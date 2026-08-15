"""Joint multi-exit training loop for IntelliDepthNet.

Trains all exits simultaneously with a weighted sum of cross-entropy losses.
Logs per-exit accuracy and saves the best model by final-exit validation accuracy.
"""

import os
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from config import TRAIN_CFG, MODEL_CFG
from datasets import get_dataloaders
from model import IntelliDepthNet
from utils import AverageMeter, accuracy, save_checkpoint, CSVLogger


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, List[float]]:
    """Run one training epoch.

    Returns:
        avg_loss: average total loss over the epoch.
        exit_accs: list of top-1 accuracies per exit.
    """
    model.train()
    loss_meter = AverageMeter()
    exit_acc_meters = [AverageMeter() for _ in range(model.num_exits)]

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits_list = model(images)
        # Weighted sum of CE losses
                # Weighted sum of CE losses + distillation from final exit
        weights = get_exit_weights_for_epoch(epoch)
        loss = 0.0
        final_logits = logits_list[-1].detach()  # Teacher: final exit (no gradient)
        
        for idx, logits in enumerate(logits_list):
            # Standard cross-entropy
            w = weights[idx] if idx < len(weights) else 1.0
            loss += w * criterion(logits, labels)
            
            # Distillation: early exits mimic final exit (skip for final itself)
            if idx < len(logits_list) - 1:
                distill_temp = 4.0  # Softmax temperature for distillation
                final_probs = F.softmax(final_logits / distill_temp, dim=1)
                current_log_probs = F.log_softmax(logits / distill_temp, dim=1)
                distill_loss = F.kl_div(current_log_probs, final_probs, reduction='batchmean')
                loss += 0.5 * distill_loss  # Weight of distillation term

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        for idx, logits in enumerate(logits_list):
            acc = accuracy(logits, labels, topk=(1,))[0]
            exit_acc_meters[idx].update(acc, images.size(0))

    exit_accs = [m.avg for m in exit_acc_meters]
    return loss_meter.avg, exit_accs


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, List[float]]:
    """Run validation.

    Returns:
        avg_loss: weighted total loss.
        exit_accs: per-exit top-1 accuracies.
    """
    model.eval()
    loss_meter = AverageMeter()
    exit_acc_meters = [AverageMeter() for _ in range(model.num_exits)]

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits_list = model(images)

        loss = 0.0
        for idx, logits in enumerate(logits_list):
            loss += criterion(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        for idx, logits in enumerate(logits_list):
            acc = accuracy(logits, labels, topk=(1,))[0]
            exit_acc_meters[idx].update(acc, images.size(0))

    exit_accs = [m.avg for m in exit_acc_meters]
    return loss_meter.avg, exit_accs

def get_exit_weights_for_epoch(
    epoch: int,
    schedule=None,
) -> List[float]:
    """Return loss weights for the current epoch based on progressive schedule.
    
    The schedule is a list of (start_epoch, end_epoch, weights). We pick the
    first phase whose range contains the current epoch.
    
    Args:
        epoch: Current training epoch (0-indexed).
        schedule: List of (start, end, weights). If None, uses config default.
    
    Returns:
        List of float weights, one per exit.
    """
    if schedule is None:
        schedule = TRAIN_CFG.exit_weight_schedule
    
    for start, end, weights in schedule:
        if start <= epoch < end:
            return weights
    
    # Fallback: return the last phase's weights
    return schedule[-1][2]

def main():
    parser = argparse.ArgumentParser(description="Train IntelliDepthNet")
    parser.add_argument("--quick_test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    quick = args.quick_test

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(quick_test=quick)

    # Model
    model = IntelliDepthNet().to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=TRAIN_CFG.lr,
        momentum=TRAIN_CFG.momentum,
        weight_decay=TRAIN_CFG.weight_decay,
    )

    epochs = TRAIN_CFG.quick_test_epochs if quick else TRAIN_CFG.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_epoch = 0
    best_final_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_final_acc = checkpoint.get("best_final_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(TRAIN_CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(TRAIN_CFG.log_dir, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "val_loss"] +                  [f"train_acc_exit{i}" for i in range(model.num_exits)] +                  [f"val_acc_exit{i}" for i in range(model.num_exits)] +                  ["lr"]
    logger = CSVLogger(os.path.join(TRAIN_CFG.log_dir, "train_log.csv"), fieldnames)

    print(f"Training for {epochs} epochs on device {device}")
    print(f"Model exits: {model.num_exits} | Progressive weight schedule active")

    for epoch in range(start_epoch, epochs):
        train_loss, train_accs = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_accs = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        final_val_acc = val_accs[-1]
        is_best = final_val_acc > best_final_acc
        if is_best:
            best_final_acc = final_val_acc

        # Logging
        row = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
        }
        for i, acc in enumerate(train_accs):
            row[f"train_acc_exit{i}"] = f"{acc:.2f}"
        for i, acc in enumerate(val_accs):
            row[f"val_acc_exit{i}"] = f"{acc:.2f}"
        logger.log(row)

        print(
            f"Epoch [{epoch+1}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
            f"Final Val Acc: {final_val_acc:.2f}%  "
            f"Best: {best_final_acc:.2f}%  LR: {optimizer.param_groups[0]['lr']:.4f}"
        )

        current_weights = get_exit_weights_for_epoch(epoch)
        print(f"  Exit weights this epoch: {current_weights}")
        # Checkpointing
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_final_acc": best_final_acc,
            "val_accs": val_accs,
        }
        if (epoch + 1) % TRAIN_CFG.save_every == 0 or epoch == epochs - 1:
            save_checkpoint(
                state, is_best=False,
                checkpoint_dir=TRAIN_CFG.checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth",
            )
        if is_best:
            save_checkpoint(
                state, is_best=True,
                checkpoint_dir=TRAIN_CFG.checkpoint_dir,
                filename="model_best.pth",
            )

    logger.close()
    print(f"Training complete. Best final-exit validation accuracy: {best_final_acc:.2f}%")


if __name__ == "__main__":
    main()
