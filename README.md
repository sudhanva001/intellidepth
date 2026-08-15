# IntelliDepth

Confidence-calibrated adaptive early-exit deep learning for image classification (PyTorch).

## Overview

IntelliDepth trains a CIFAR-style ResNet with multiple lightweight exit heads. During inference, each sample exits as early as possible when the **calibrated** confidence exceeds a threshold. Temperature scaling is applied independently to every exit using a held-out validation set, ensuring that confidence scores are reliable and not overconfident — a common failure mode of raw softmax scores at shallow layers.

## Project Structure

```
intellidepth/
├── config.py               # All hyperparameters in one place
├── datasets.py             # CIFAR-100 + CIFAR-100-C loaders
├── model.py                # IntelliDepthNet (backbone + exit heads)
├── train.py                # Joint multi-exit training
├── calibration.py          # Temperature scaling + ECE
├── adaptive_inference.py   # Early-exit policy + threshold search
├── evaluate.py             # Plots, robustness tests, summary tables
├── utils.py                # Logging, checkpointing, metrics
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start (Smoke Test)

Run a 1-epoch quick test on data subsets to verify the pipeline:

```bash
# 1. Train (quick test)
python train.py --quick_test

# 2. Calibrate
python calibration.py --checkpoint checkpoints/model_best.pth --quick_test

# 3. Evaluate
python evaluate.py --checkpoint checkpoints/model_best.pth --quick_test
```

## Full Pipeline

### 1. Train the backbone

```bash
python train.py
```

- Trains ResNet-56 with 4 exits on CIFAR-100 (45k train / 5k val).
- Saves best model to `checkpoints/model_best.pth`.
- Logs per-exit accuracy every epoch to `logs/train_log.csv`.

### 2. Calibrate exit temperatures

```bash
python calibration.py --checkpoint checkpoints/model_best.pth
```

- Fits a scalar temperature `T_i` per exit on the held-out validation set.
- Prints ECE before/after for each exit.
- Saves `checkpoints/temperatures.json`.

### 3. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/model_best.pth
```

Generates in `plots/`:
- `accuracy_vs_compute.png` — tradeoff curve (calibrated vs raw vs baseline)
- `ece_per_exit.png` — ECE before/after bar chart
- `exit_distribution.png` — histogram of exit points
- `reliability_exit{i}_before.png` / `reliability_exit{i}_after.png` — reliability diagrams
- `robustness.png` — CIFAR-100-C corruption robustness

And writes `results_summary.csv` with all numeric results.

## Adaptive Inference API

```python
from model import IntelliDepthNet
from adaptive_inference import AdaptiveInference
from calibration import load_json

model = IntelliDepthNet()
model.load_state_dict(...)

temperatures = load_json("checkpoints/temperatures.json")
temps = [temperatures[f"exit_{i}"] for i in range(model.num_exits)]

engine = AdaptiveInference(model, temps, device="cuda")

# Fixed threshold
metrics = engine.evaluate(test_loader, thresholds=[0.9]*4, use_calibration=True)
print(metrics["accuracy"], metrics["avg_compute_ratio"])

# Grid search for target compute reduction
best = engine.threshold_grid_search(val_loader, target_compute_reductions=[0.3, 0.4])
```

## Key Design Decisions

- **CIFAR-style ResNet**: 3×3 stem, no maxpool, 3 stages. This is the standard architecture for CIFAR, different from ImageNet ResNets.
- **Exit heads**: One 3×3 conv + GAP + FC. Kept intentionally lightweight so early-exit savings are not eaten by heavy heads.
- **Temperature scaling per exit**: Each exit has its own `T_i`. Shallow exits are typically more overconfident and benefit more from scaling.
- **`forward_single_exit`**: During adaptive inference, deeper layers are skipped entirely. This is essential for real latency reduction, not just theoretical FLOP counts.
- **Configurable exits**: The number and placement of exits is controlled by `exit_stage_indices` in `config.py`.

## CIFAR-100-C Setup

Download CIFAR-100-C from [hendrycks/robustness](https://github.com/hendrycks/robustness) and extract to:

```
./data/CIFAR-100-C/
    labels.npy
    fog.npy
    gaussian_noise.npy
    motion_blur.npy
    brightness.npy
    ...
```

The evaluation script will automatically pick up the corruptions specified in `config.py`.

## Customization

Edit `config.py` to change:
- `n_blocks_per_stage`: 9 for ResNet-56, 5 for ResNet-32
- `exit_stage_indices`: where to attach intermediate exits
- `exit_loss_weights`: weighting of each exit during training
- `global_threshold`, `target_flops_reductions`: inference behavior

## Citation

If you use this code, please cite the relevant works on early-exit networks and temperature scaling (Guo et al., 2017).
