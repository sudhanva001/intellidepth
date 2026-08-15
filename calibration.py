"""Temperature scaling calibration and Expected Calibration Error (ECE).

After training, each exit head is calibrated independently on the held-out
validation set.  A scalar temperature T_i is learned per exit to minimize
negative log-likelihood, making softmax confidences better reflect true
probabilities.  This is crucial for adaptive early-exit: without calibration,
intermediate exits tend to be overconfident, causing premature (and wrong) exits.
"""

import os
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import CALIB_CFG, TRAIN_CFG, MODEL_CFG
from datasets import get_dataloaders
from model import IntelliDepthNet
from utils import save_json, load_json


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = CALIB_CFG.n_bins,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins.

    ECE = sum_{bin} (n_bin / N) * |acc_bin - conf_bin|

    Args:
        probs: Array of shape (N, C) with softmax probabilities.
        labels: Array of shape (N,) with integer class labels.
        n_bins: Number of confidence bins.

    Returns:
        ECE as a float in [0, 1].
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = probs.shape[0]

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > low) & (confidences <= high)
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)

        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = CALIB_CFG.max_iter,
    lr: float = CALIB_CFG.lr,
    temperature_init: float = CALIB_CFG.temperature_init,
) -> float:
    """Learn a scalar temperature that minimizes NLL on validation logits.

    We use LBFGS because temperature scaling is a single-parameter convex
    problem; LBFGS converges in very few iterations and is the standard
    choice in the literature (Guo et al., "On Calibration of Modern NN").

    Args:
        logits: Tensor of shape (N, C) from a single exit.
        labels: Tensor of shape (N,) with ground-truth class indices.
        max_iter: Maximum LBFGS iterations.
        lr: Learning rate / step size for LBFGS.
        temperature_init: Initial temperature (>0).

    Returns:
        Optimized scalar temperature.
    """
    temperature = nn.Parameter(torch.tensor([temperature_init], device=logits.device))
    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def eval_loss():
        optimizer.zero_grad()
        # Temperature must stay positive; we optimize in log-space implicitly
        # by taking abs, or simply clamp.  Here we use softplus for smoothness.
        t = F.softplus(temperature) + 1e-6
        scaled_logits = logits / t
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    t_opt = (F.softplus(temperature) + 1e-6).item()
    return t_opt


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits (B, C) or (N, C).
        temperature: Positive scalar temperature.

    Returns:
        Calibrated probabilities after softmax(logits / T).
    """
    return F.softmax(logits / temperature, dim=1)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run inference and collect logits from every exit.

    Returns:
        all_logits: List of tensors, one per exit, each shape (N, C).
        all_labels: Tensor of shape (N,).
    """
    model.eval()
    all_logits_per_exit: List[List[torch.Tensor]] = [[] for _ in range(model.num_exits)]
    all_labels: List[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device)
        logits_list = model(images)
        for idx, logits in enumerate(logits_list):
            all_logits_per_exit[idx].append(logits.cpu())
        all_labels.append(labels)

    all_logits = [torch.cat(lst, dim=0) for lst in all_logits_per_exit]
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels


def calibrate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = os.path.join(TRAIN_CFG.checkpoint_dir, "temperatures.json"),
) -> Tuple[List[float], List[float], List[float]]:
    """Calibrate every exit of the model and report ECE before/after.

    The backbone is frozen; only temperatures are learned.

    Returns:
        temperatures: List of T_i per exit.
        ece_before: List of ECE values before calibration.
        ece_after: List of ECE values after calibration.
    """
    print("Collecting validation logits...")
    all_logits, all_labels = collect_logits(model, val_loader, device)

    temperatures = []
    ece_before = []
    ece_after = []

    for exit_idx in range(model.num_exits):
        logits = all_logits[exit_idx]
        labels = all_labels.numpy()

        # Before calibration
        probs_before = F.softmax(logits, dim=1).numpy()
        ece_b = compute_ece(probs_before, labels)
        ece_before.append(ece_b)

        # Fit temperature
        print(f"  Exit {exit_idx}: fitting temperature...")
        t = fit_temperature(logits.to(device), all_labels.to(device))
        temperatures.append(t)

        # After calibration
        probs_after = apply_temperature(logits, t).numpy()
        ece_a = compute_ece(probs_after, labels)
        ece_after.append(ece_a)

        print(
            f"  Exit {exit_idx}: T={t:.4f} | ECE before={ece_b:.4f} | ECE after={ece_a:.4f}"
        )

    # Save temperatures
    save_json({f"exit_{i}": t for i, t in enumerate(temperatures)}, save_path)
    print(f"Temperatures saved to {save_path}")

    return temperatures, ece_before, ece_after


def main():
    parser = argparse.ArgumentParser(description="Calibrate IntelliDepth exits")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    _, val_loader, _ = get_dataloaders(quick_test=args.quick_test)

    model = IntelliDepthNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze backbone and heads so only temperature is learned
    for param in model.parameters():
        param.requires_grad = False

    calibrate_model(model, val_loader, device)


if __name__ == "__main__":
    main()
