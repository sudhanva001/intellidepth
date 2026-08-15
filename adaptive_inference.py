"""Adaptive early-exit inference engine.

At test time, samples traverse the network stage by stage.  After each
stage the corresponding exit head produces logits, which are temperature-scaled
to calibrated confidence scores.  If the max confidence exceeds a threshold,
inference stops immediately — saving the compute of deeper layers.

This module implements both fixed-threshold and per-exit threshold-search
policies, plus evaluation metrics.
"""

import os
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from config import INFER_CFG, TRAIN_CFG, MODEL_CFG
from datasets import get_dataloaders
from model import IntelliDepthNet
from calibration import apply_temperature, load_json
from utils import accuracy


class AdaptiveInference:
    """Runs adaptive early-exit inference on a trained and calibrated model."""

    def __init__(
        self,
        model: nn.Module,
        temperatures: List[float],
        device: torch.device,
    ):
        """
        Args:
            model: Trained IntelliDepthNet.
            temperatures: List of scalar temperatures, one per exit.
            device: torch device.
        """
        self.model = model
        self.model.eval()
        self.temperatures = temperatures
        self.device = device
        self.num_exits = model.num_exits

        # Pre-compute relative FLOPs ratios for each exit
        self.exit_costs = [model.get_exit_flops_ratio(i) for i in range(self.num_exits)]

    @torch.no_grad()
    def predict_single(
        self,
        x: torch.Tensor,
        thresholds: List[float],
        use_calibration: bool = True,
    ) -> Tuple[int, torch.Tensor, float]:
        """Run adaptive inference on a single sample.

        Args:
            x: Input tensor of shape (1, C, H, W) or (C, H, W).
            thresholds: List of confidence thresholds, one per exit.
            use_calibration: If True, apply temperature scaling.

        Returns:
            exit_idx: Index of exit that fired (0-based).
            logits: Logits from that exit.
            confidence: Max softmax confidence that triggered the exit.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        for exit_idx in range(self.num_exits):
            logits = self.model.forward_single_exit(x, exit_idx)
            if use_calibration:
                probs = apply_temperature(logits, self.temperatures[exit_idx])
            else:
                probs = torch.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values.item()

            if confidence >= thresholds[exit_idx]:
                return exit_idx, logits, confidence

        # Should always reach final exit, but fallback
        return self.num_exits - 1, logits, confidence

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        thresholds: List[float],
        use_calibration: bool = True,
    ) -> Dict[str, float]:
        """Evaluate adaptive policy on a dataset.

        Returns:
            Dictionary with keys:
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

            for b in range(batch_size):
                exit_idx, logits, _ = self.predict_single(
                    images[b], thresholds, use_calibration
                )
                pred = logits.argmax(dim=1)
                correct += (pred == labels[b]).item()
                exit_counts[exit_idx] += 1
                total_cost += self.exit_costs[exit_idx]

            total += batch_size

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

    def threshold_grid_search(
        self,
        val_loader: torch.utils.data.DataLoader,
        target_compute_reductions: List[float] = INFER_CFG.target_flops_reductions,
        grid: List[float] = INFER_CFG.threshold_search_grid,
        use_calibration: bool = True,
    ) -> Dict[float, List[float]]:
        """Search for per-exit thresholds that achieve target compute reductions.

        We perform a simple greedy grid search: for each target, we try
        all combinations of thresholds from the grid and pick the one that
        achieves the closest compute reduction while maximizing accuracy.
        To keep search tractable, we use the same threshold for all exits
        within a candidate, but you can extend this to independent thresholds.

        Args:
            val_loader: Validation dataloader (held-out calibration set).
            target_compute_reductions: Target fractions of compute to save.
            grid: Candidate threshold values.
            use_calibration: Whether to use calibrated confidences.

        Returns:
            Mapping target_reduction -> best_thresholds list.
        """
        print("Running threshold grid search on validation set...")
        results = {}

        # Collect all samples' confidences per exit for smarter search
        all_confs = [[] for _ in range(self.num_exits)]
        all_labels_list = []
        for images, labels in val_loader:
            images = images.to(self.device)
            for b in range(images.size(0)):
                for exit_idx in range(self.num_exits):
                    logits = self.model.forward_single_exit(images[b].unsqueeze(0), exit_idx)
                    if use_calibration:
                        probs = apply_temperature(logits, self.temperatures[exit_idx])
                    else:
                        probs = torch.softmax(logits, dim=1)
                    conf = probs.max(dim=1).values.item()
                    all_confs[exit_idx].append(conf)
            all_labels_list.extend(labels.tolist())

        # For simplicity, use uniform threshold across exits
        best_for_target = {}
        for target in target_compute_reductions:
            best_acc = -1.0
            best_thresh = None
            best_metrics = None
            target_ratio = 1.0 - target  # e.g., 0.6 means use 60% of full compute

            for tau in grid:
                thresholds = [tau] * self.num_exits
                metrics = self.evaluate(val_loader, thresholds, use_calibration)
                compute_ratio = metrics["avg_compute_ratio"]
                # Pick the threshold whose compute ratio is closest to target
                # and has highest accuracy among ties
                if abs(compute_ratio - target_ratio) < abs(
                    best_metrics["avg_compute_ratio"] - target_ratio
                ) if best_metrics else True:
                    best_metrics = metrics
                    best_thresh = thresholds
                elif best_metrics and abs(compute_ratio - target_ratio) == abs(
                    best_metrics["avg_compute_ratio"] - target_ratio
                ):
                    if metrics["accuracy"] > best_metrics["accuracy"]:
                        best_metrics = metrics
                        best_thresh = thresholds

            results[target] = best_thresh
            print(
                f"  Target reduction {target:.0%}: thresholds={best_thresh}, "
                f"compute={best_metrics['avg_compute_ratio']:.3f}, acc={best_metrics['accuracy']:.2f}%"
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="Adaptive inference demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--temperatures", type=str, default=None,
                        help="Path to temperatures.json (auto-detected if None)")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
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

    # Fixed global threshold
    global_tau = INFER_CFG.global_threshold
    print(f"\n=== Fixed threshold τ={global_tau} ===")
    for name, loader in [("Validation", val_loader), ("Test", test_loader)]:
        metrics_calib = engine.evaluate(loader, [global_tau] * model.num_exits, use_calibration=True)
        metrics_raw = engine.evaluate(loader, [global_tau] * model.num_exits, use_calibration=False)
        print(f"  {name} (calibrated):   acc={metrics_calib['accuracy']:.2f}%, "
              f"avg_compute={metrics_calib['avg_compute_ratio']:.3f}")
        print(f"  {name} (raw conf):     acc={metrics_raw['accuracy']:.2f}%, "
              f"avg_compute={metrics_raw['avg_compute_ratio']:.3f}")

    # Grid search
    print("\n=== Threshold grid search ===")
    best_thresholds = engine.threshold_grid_search(val_loader)
    print("\nBest thresholds per target:")
    for target, thresh in best_thresholds.items():
        print(f"  {target:.0%} reduction -> {thresh}")


if __name__ == "__main__":
    main()
