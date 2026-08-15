"""Fast batched evaluation for IntelliDepth.

Uses vectorized adaptive inference: processes the entire batch through each stage,
marking samples as "done" when they exit, so GPU parallelism is maintained.
Runtime: ~30-60 seconds for full CIFAR-100 test set on GPU.
"""

import os
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EVAL_CFG, TRAIN_CFG, INFER_CFG, MODEL_CFG
from datasets import get_dataloaders
from model import IntelliDepthNet
from calibration import apply_temperature, compute_ece, collect_logits, load_json
from utils import accuracy


class BatchedAdaptiveInference:
    """Batched adaptive early-exit inference engine.

    Unlike the per-sample loop in adaptive_inference.py, this processes
    entire batches through each stage. Samples that haven't exited yet
    continue to the next stage together, maintaining GPU parallelism.
    """

    def __init__(self, model: nn.Module, temperatures: List[float], device: torch.device):
        self.model = model
        self.model.eval()
        self.temperatures = temperatures
        self.device = device
        self.num_exits = model.num_exits
        self.exit_costs = [model.get_exit_flops_ratio(i) for i in range(self.num_exits)]

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        thresholds: List[float],
        use_calibration: bool = True,
    ) -> Dict[str, float]:
        """Batched adaptive evaluation.

        Returns:
            accuracy, avg_exit_idx, avg_compute_ratio, exit_distribution
        """
        correct = 0
        total = 0
        exit_counts = [0] * self.num_exits
        total_cost = 0.0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Track which samples are still active and their predictions
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            exit_depths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # Stage-by-stage batched processing
            # We run the full backbone forward but check exits stage by stage
            # This is slightly less optimal than true early-exit (some wasted compute
            # for samples that exit early), but much faster than per-sample loops.
            # For true compute savings, use forward_single_exit in deployment.

            all_logits = self.model(images)  # List of logits per exit

            for exit_idx in range(self.num_exits):
                if not active_mask.any():
                    break

                logits = all_logits[exit_idx]
                if use_calibration:
                    probs = apply_temperature(logits, self.temperatures[exit_idx])
                else:
                    probs = F.softmax(logits, dim=1)

                confidences, preds = probs.max(dim=1)

                # For active samples, check if they meet threshold
                can_exit = active_mask & (confidences >= thresholds[exit_idx])

                # Record predictions for samples exiting here
                predictions[can_exit] = preds[can_exit]
                exit_depths[can_exit] = exit_idx
                active_mask[can_exit] = False

                # If last exit, force remaining active samples to exit
                if exit_idx == self.num_exits - 1:
                    predictions[active_mask] = preds[active_mask]
                    exit_depths[active_mask] = exit_idx
                    active_mask[:] = False

            # Compute accuracy
            correct += (predictions == labels).sum().item()
            total += batch_size

            for i in range(self.num_exits):
                count = (exit_depths == i).sum().item()
                exit_counts[i] += count
                total_cost += count * self.exit_costs[i]

        accuracy_pct = 100.0 * correct / total
        avg_exit_idx = sum(i * c for i, c in enumerate(exit_counts)) / total
        avg_compute_ratio = total_cost / total
        exit_distribution = [c / total for c in exit_counts]

        return {
            "accuracy": accuracy_pct,
            "avg_exit_idx": avg_exit_idx,
            "avg_compute_ratio": avg_compute_ratio,
            "exit_distribution": exit_distribution,
            "exit_counts": exit_counts,
        }


def plot_accuracy_vs_compute(calib_accs, calib_costs, raw_accs, raw_costs, baseline_acc, plot_dir):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(calib_costs, calib_accs, "o-", label="Calibrated confidence", color="tab:blue", linewidth=2)
    ax.plot(raw_costs, raw_accs, "s--", label="Raw confidence", color="tab:orange", linewidth=2)
    ax.axhline(baseline_acc, color="tab:gray", linestyle=":", label="Final-exit baseline", linewidth=2)
    ax.set_xlabel("Average compute ratio (relative to full network)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Compute Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "accuracy_vs_compute.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Saved: {p}")


def plot_ece_per_exit(ece_before, ece_after, plot_dir):
    x = np.arange(len(ece_before))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(x - width/2, ece_before, width, label="Before calibration", color="tab:red", alpha=0.8)
    ax.bar(x + width/2, ece_after, width, label="After calibration", color="tab:green", alpha=0.8)
    ax.set_xlabel("Exit index", fontsize=12)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=12)
    ax.set_title("Per-Exit Calibration Quality", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Exit {i}" for i in range(len(ece_before))])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(plot_dir, "ece_per_exit.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Saved: {p}")


def plot_exit_distribution(dist, thresholds, plot_dir):
    fig, ax = plt.subplots(figsize=(7,5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dist)))
    bars = ax.bar(range(len(dist)), dist, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Exit index", fontsize=12)
    ax.set_ylabel("Fraction of samples", fontsize=12)
    ax.set_title(f"Exit Distribution (thresholds={thresholds})", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels([f"Exit {i}" for i in range(len(dist))])
    for bar, val in zip(bars, dist):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(plot_dir, "exit_distribution.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser(description="Fast batched evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperatures", type=str, default=None)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    os.makedirs(EVAL_CFG.plot_dir, exist_ok=True)

    _, val_loader, test_loader = get_dataloaders(quick_test=args.quick_test)

    model = IntelliDepthNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    temps_path = args.temperatures or os.path.join(TRAIN_CFG.checkpoint_dir, "temperatures.json")
    temps_dict = load_json(temps_path)
    temperatures = [temps_dict[f"exit_{i}"] for i in range(model.num_exits)]
    print(f"Loaded temperatures: {temperatures}")

    engine = BatchedAdaptiveInference(model, temperatures, device)

    # Collect logits for ECE computation
    print("\nCollecting validation logits for ECE...")
    all_logits, all_labels = collect_logits(model, val_loader, device)

    ece_before, ece_after = [], []
    for i in range(model.num_exits):
        ece_b = compute_ece(torch.softmax(all_logits[i], dim=1).cpu().numpy(), all_labels.numpy())
        ece_a = compute_ece(apply_temperature(all_logits[i], temperatures[i]).cpu().numpy(), all_labels.numpy())
        ece_before.append(ece_b)
        ece_after.append(ece_a)

    print("\n=== CALIBRATION SUMMARY ===")
    for i in range(model.num_exits):
        print(f"  Exit {i}: ECE before={ece_before[i]:.4f} | after={ece_after[i]:.4f} | T={temperatures[i]:.4f}")

    # Baseline: always final exit
    print("\nEvaluating final-exit baseline...")
    baseline_metrics = engine.evaluate(test_loader, [2.0]*model.num_exits, use_calibration=True)
    baseline_acc = baseline_metrics["accuracy"]
    print(f"  Baseline (always final): {baseline_acc:.2f}% accuracy, compute=1.000")

    # Threshold sweep
    print("\nRunning threshold sweep on test set...")
    threshold_grid = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    calib_accs, calib_costs = [], []
    raw_accs, raw_costs = [], []

    for tau in threshold_grid:
        th = [tau] * model.num_exits
        mc = engine.evaluate(test_loader, th, use_calibration=True)
        mr = engine.evaluate(test_loader, th, use_calibration=False)
        calib_accs.append(mc["accuracy"])
        calib_costs.append(mc["avg_compute_ratio"])
        raw_accs.append(mr["accuracy"])
        raw_costs.append(mr["avg_compute_ratio"])
        print(f"  τ={tau:.2f}: Calib={mc['accuracy']:.2f}% @ {mc['avg_compute_ratio']:.3f} | Raw={mr['accuracy']:.2f}% @ {mr['avg_compute_ratio']:.3f}")

    # Grid search on validation
    print("\nRunning grid search on validation set...")
    val_engine = BatchedAdaptiveInference(model, temperatures, device)
    best_thresholds = {}
    for target in INFER_CFG.target_flops_reductions:
        target_ratio = 1.0 - target
        best_acc = -1.0
        best_tau = None
        best_compute = None
        for tau in INFER_CFG.threshold_search_grid:
            th = [tau] * model.num_exits
            m = val_engine.evaluate(val_loader, th, use_calibration=True)
            if abs(m["avg_compute_ratio"] - target_ratio) < 0.05 and m["accuracy"] > best_acc:
                best_acc = m["accuracy"]
                best_tau = tau
                best_compute = m["avg_compute_ratio"]
        if best_tau is None:
            best_tau = 0.9
            m = val_engine.evaluate(val_loader, [best_tau]*model.num_exits, True)
            best_compute = m["avg_compute_ratio"]
            best_acc = m["accuracy"]
        best_thresholds[target] = [best_tau] * model.num_exits
        print(f"  Target {target:.0%} reduction: τ={best_tau:.2f}, compute={best_compute:.3f}, acc={best_acc:.2f}%")

    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_vs_compute(calib_accs, calib_costs, raw_accs, raw_costs, baseline_acc, EVAL_CFG.plot_dir)
    plot_ece_per_exit(ece_before, ece_after, EVAL_CFG.plot_dir)

    # Exit distribution at best threshold (40% reduction target)
    best_t = best_thresholds.get(0.4, [0.9]*model.num_exits)
    dist_metrics = engine.evaluate(test_loader, best_t, use_calibration=True)
    plot_exit_distribution(dist_metrics["exit_distribution"], best_t, EVAL_CFG.plot_dir)

    # Save CSV
    lines = ["threshold,calib_acc,raw_acc,compute_calib,compute_raw"]
    for tau, ca, ra, cc, rc in zip(threshold_grid, calib_accs, raw_accs, calib_costs, raw_costs):
        lines.append(f"{tau:.2f},{ca:.2f},{ra:.2f},{cc:.4f},{rc:.4f}")
    lines.append("")
    lines.append("exit_idx,uncalibrated_ece,calibrated_ece")
    for i in range(model.num_exits):
        lines.append(f"{i},{ece_before[i]:.4f},{ece_after[i]:.4f}")

    with open(EVAL_CFG.results_csv, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to {EVAL_CFG.results_csv}")
    print("Plots saved to", EVAL_CFG.plot_dir)
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
