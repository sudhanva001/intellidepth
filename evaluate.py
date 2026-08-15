"""Comprehensive evaluation and plotting for IntelliDepth.

Produces:
  1. Accuracy vs. Compute tradeoff curves (calibrated vs raw vs baseline).
  2. ECE bar chart per exit, before/after calibration.
  3. Exit distribution histograms.
  4. Robustness evaluation on CIFAR-100-C.
  5. Reliability diagrams per exit.
  6. Summary CSV with all metrics.
"""

import os
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EVAL_CFG, TRAIN_CFG, INFER_CFG, MODEL_CFG
from datasets import get_dataloaders, get_corrupted_loader
from model import IntelliDepthNet
from calibration import (
    apply_temperature, compute_ece, collect_logits, load_json,
)
from adaptive_inference import AdaptiveInference
from utils import accuracy, save_json


def plot_accuracy_vs_compute(
    engine: AdaptiveInference,
    test_loader: torch.utils.data.DataLoader,
    thresholds_grid: List[float],
    plot_dir: str,
) -> None:
    """Plot accuracy vs average compute ratio for calibrated and raw policies."""
    calib_accs, calib_costs = [], []
    raw_accs, raw_costs = [], []

    for tau in thresholds_grid:
        thresholds = [tau] * engine.num_exits
        m_calib = engine.evaluate(test_loader, thresholds, use_calibration=True)
        m_raw = engine.evaluate(test_loader, thresholds, use_calibration=False)
        calib_accs.append(m_calib["accuracy"])
        calib_costs.append(m_calib["avg_compute_ratio"])
        raw_accs.append(m_raw["accuracy"])
        raw_costs.append(m_raw["avg_compute_ratio"])

    # Baseline: always final exit
    baseline_acc = engine.evaluate(test_loader, [2.0] * engine.num_exits, use_calibration=True)["accuracy"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(calib_costs, calib_accs, "o-", label="Calibrated confidence", color="tab:blue")
    ax.plot(raw_costs, raw_accs, "s--", label="Raw confidence", color="tab:orange")
    ax.axhline(baseline_acc, color="tab:gray", linestyle=":", label="Final-exit baseline")
    ax.set_xlabel("Average compute ratio (relative to full network)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs. Compute Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "accuracy_vs_compute.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ece_per_exit(
    ece_before: List[float],
    ece_after: List[float],
    plot_dir: str,
) -> None:
    """Bar chart of ECE before vs after calibration for each exit."""
    x = np.arange(len(ece_before))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, ece_before, width, label="Before calibration", color="tab:red", alpha=0.8)
    ax.bar(x + width / 2, ece_after, width, label="After calibration", color="tab:green", alpha=0.8)
    ax.set_xlabel("Exit index")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Per-Exit Calibration Quality")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Exit {i}" for i in range(len(ece_before))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(plot_dir, "ece_per_exit.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_exit_distribution(
    engine: AdaptiveInference,
    test_loader: torch.utils.data.DataLoader,
    thresholds: List[float],
    plot_dir: str,
) -> None:
    """Histogram of how many samples exit at each point."""
    metrics = engine.evaluate(test_loader, thresholds, use_calibration=True)
    dist = metrics["exit_distribution"]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dist)))
    bars = ax.bar(range(len(dist)), dist, color=colors)
    ax.set_xlabel("Exit index")
    ax.set_ylabel("Fraction of samples")
    ax.set_title(f"Exit Distribution (thresholds={thresholds})")
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels([f"Exit {i}" for i in range(len(dist))])
    for bar, val in zip(bars, dist):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(plot_dir, "exit_distribution.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    save_path: str,
    n_bins: int = 15,
) -> None:
    """Plot a reliability diagram (confidence vs accuracy per bin)."""
    probs = torch.softmax(logits, dim=1).numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels.numpy()).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > low) & (confidences <= high)
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        if in_bin.sum() > 0:
            bin_accs[i] = accuracies[in_bin].mean()
            bin_confs[i] = confidences[in_bin].mean()
            bin_counts[i] = in_bin.sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6, edgecolor="black", label="Accuracy")
    # Show gap as error bars
    gaps = np.abs(bin_confs - bin_accs)
    ax.bar(bin_centers, gaps, bottom=np.minimum(bin_accs, bin_confs),
           width=1.0 / n_bins, alpha=0.3, color="red", label="Gap")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def evaluate_robustness(
    engine: AdaptiveInference,
    corruption_types: List[str],
    severities: List[int],
    global_threshold: float,
    plot_dir: str,
) -> Dict[str, Dict]:
    """Evaluate on CIFAR-100-C and compare calibrated vs raw policies."""
    results = {}
    for ctype in corruption_types:
        for sev in severities:
            try:
                loader = get_corrupted_loader(ctype, sev)
            except FileNotFoundError as e:
                print(f"  Skipping {ctype} severity {sev}: {e}")
                continue

            thresholds = [global_threshold] * engine.num_exits
            m_calib = engine.evaluate(loader, thresholds, use_calibration=True)
            m_raw = engine.evaluate(loader, thresholds, use_calibration=False)

            key = f"{ctype}_sev{sev}"
            results[key] = {
                "calib_acc": m_calib["accuracy"],
                "raw_acc": m_raw["accuracy"],
                "calib_compute": m_calib["avg_compute_ratio"],
                "raw_compute": m_raw["avg_compute_ratio"],
            }
            print(
                f"  {key}: Calib={m_calib['accuracy']:.2f}% Raw={m_raw['accuracy']:.2f}%"
            )

    # Plot robustness summary
    if results:
        keys = list(results.keys())
        calib_accs = [results[k]["calib_acc"] for k in keys]
        raw_accs = [results[k]["raw_acc"] for k in keys]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.5), 5))
        ax.plot(x, calib_accs, "o-", label="Calibrated", color="tab:blue")
        ax.plot(x, raw_accs, "s--", label="Raw confidence", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Robustness: CIFAR-100-C")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, "robustness.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate IntelliDepth")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperatures", type=str, default=None)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(EVAL_CFG.plot_dir, exist_ok=True)

    _, val_loader, test_loader = get_dataloaders(quick_test=args.quick_test)

    model = IntelliDepthNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    temps_path = args.temperatures
    if temps_path is None:
        temps_path = os.path.join(TRAIN_CFG.checkpoint_dir, "temperatures.json")
    temps_dict = load_json(temps_path)
    temperatures = [temps_dict[f"exit_{i}"] for i in range(model.num_exits)]

    engine = AdaptiveInference(model, temperatures, device)

    # ------------------------------------------------------------------
    # 1. Collect logits for calibration visualization
    # ------------------------------------------------------------------
    print("Collecting validation logits for reliability diagrams...")
    all_logits, all_labels = collect_logits(model, val_loader, device)

    # Compute ECE before/after for table
    ece_before = []
    ece_after = []
    for i in range(model.num_exits):
        probs_b = torch.softmax(all_logits[i], dim=1).numpy()
        ece_b = compute_ece(probs_b, all_labels.numpy())
        probs_a = apply_temperature(all_logits[i], temperatures[i]).numpy()
        ece_a = compute_ece(probs_a, all_labels.numpy())
        ece_before.append(ece_b)
        ece_after.append(ece_a)

    # ------------------------------------------------------------------
    # 2. Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    # Accuracy vs Compute
    plot_accuracy_vs_compute(
        engine, test_loader,
        thresholds_grid=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        plot_dir=EVAL_CFG.plot_dir,
    )

    # ECE per exit
    plot_ece_per_exit(ece_before, ece_after, plot_dir=EVAL_CFG.plot_dir)

    # Exit distribution (using global threshold)
    plot_exit_distribution(
        engine, test_loader,
        thresholds=[INFER_CFG.global_threshold] * engine.num_exits,
        plot_dir=EVAL_CFG.plot_dir,
    )

    # Reliability diagrams
    for i in range(model.num_exits):
        plot_reliability_diagram(
            all_logits[i], all_labels,
            title=f"Exit {i} - Before Calibration",
            save_path=os.path.join(EVAL_CFG.plot_dir, f"reliability_exit{i}_before.png"),
        )
        plot_reliability_diagram(
            all_logits[i] / temperatures[i], all_labels,
            title=f"Exit {i} - After Calibration",
            save_path=os.path.join(EVAL_CFG.plot_dir, f"reliability_exit{i}_after.png"),
        )
    print("Reliability diagrams saved.")

    # Robustness
    print("\nEvaluating robustness on CIFAR-100-C...")
    robustness_results = evaluate_robustness(
        engine,
        corruption_types=EVAL_CFG.corruption_types,
        severities=EVAL_CFG.corruption_severities,
        global_threshold=INFER_CFG.global_threshold,
        plot_dir=EVAL_CFG.plot_dir,
    )

    # ------------------------------------------------------------------
    # 3. Summary table / CSV
    # ------------------------------------------------------------------
    print("\n=== RESULTS SUMMARY ===")
    summary_lines = []
    summary_lines.append("exit_idx,uncalibrated_ece,calibrated_ece")
    for i in range(model.num_exits):
        line = f"{i},{ece_before[i]:.4f},{ece_after[i]:.4f}"
        summary_lines.append(line)
        print(f"Exit {i}: ECE before={ece_before[i]:.4f} | after={ece_after[i]:.4f}")

    # Threshold sweep accuracies
    print("\nThreshold sweep on test set:")
    summary_lines.append("")
    summary_lines.append("threshold,test_accuracy_calib,test_accuracy_raw,avg_compute_calib,avg_compute_raw")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        th = [tau] * engine.num_exits
        m_c = engine.evaluate(test_loader, th, use_calibration=True)
        m_r = engine.evaluate(test_loader, th, use_calibration=False)
        line = f"{tau:.2f},{m_c['accuracy']:.2f},{m_r['accuracy']:.2f},{m_c['avg_compute_ratio']:.4f},{m_r['avg_compute_ratio']:.4f}"
        summary_lines.append(line)
        print(f"  τ={tau:.2f}: Calib Acc={m_c['accuracy']:.2f}%  Raw Acc={m_r['accuracy']:.2f}%  "
              f"Compute={m_c['avg_compute_ratio']:.3f}")

    csv_path = EVAL_CFG.results_csv
    with open(csv_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary CSV saved to {csv_path}")

    # Also save robustness results as JSON
    if robustness_results:
        save_json(robustness_results, os.path.join(EVAL_CFG.plot_dir, "robustness_results.json"))

    print("\nEvaluation complete. All artifacts saved to", EVAL_CFG.plot_dir)


if __name__ == "__main__":
    main()
