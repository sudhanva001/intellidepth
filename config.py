"""Central configuration for IntelliDepth.

All hyperparameters live here so the rest of the codebase stays clean
and experiments are reproducible by editing a single file.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Dataset and dataloader settings."""
    dataset: str = "CIFAR100"
    batch_size: int = 128
    num_workers: int = 4
    train_val_split: float = 0.9  # 45k train / 5k val from 50k train set
    # CIFAR-100 mean/std
    mean: tuple = (0.5071, 0.4867, 0.4408)
    std: tuple = (0.2675, 0.2565, 0.2761)


@dataclass
class ModelConfig:
    """Backbone and early-exit architecture settings."""
    # ResNet depth: 6n + 2.  n=9 -> ResNet-56, n=5 -> ResNet-32
    n_blocks_per_stage: int = 9  # ResNet-56
    base_channels: int = 16
    num_classes: int = 100
    # Exit configuration: list of stage indices where exits are attached.
    # Stage 0 = after stem, Stage 1 = after stage 1, etc.
    # By default we place exits after stages 1, 2, 3 (before final pool).
    # The final classifier is always present and NOT in this list.
    exit_stage_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    # Lightweight head: 1 conv (3x3) + GAP + FC
    head_hidden_channels: int = 128


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    # LR drops at these fractions of total epochs
    # Loss weights per exit.  Length must match len(exit_stage_indices) + 1 (final).
    # By default equal weighting; deeper exits can be up-weighted.
    exit_weight_schedule: List[tuple[int, int, List[float]]] = field(default_factory=lambda: [
    (0,   60,  [3.0, 2.0, 1.5, 1.0]),   # Phase 1: early exits learn aggressively
    (60,  120, [2.0, 1.5, 1.2, 1.0]),   # Phase 2: transition to balanced
    (120, 999, [1.0, 1.0, 1.0, 1.0]),   # Phase 3: fine-tune everything equally
        ])
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 20  # save periodic checkpoint every N epochs
    # Quick-test overrides
    quick_test_epochs: int = 1
    quick_test_subset: int = 500  # number of training samples to use


@dataclass
class CalibrationConfig:
    """Temperature-scaling calibration settings."""
    n_bins: int = 15
    max_iter: int = 50
    lr: float = 0.01  # for LBFGS this is the line-search step size hint
    temperature_init: float = 1.5


@dataclass
class InferenceConfig:
    """Adaptive inference settings."""
    # Global fixed threshold baseline
    global_threshold: float = 0.9
    # Grid-search targets
    target_flops_reductions: List[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.4, 0.5]
    )
    threshold_search_grid: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )


@dataclass
class EvalConfig:
    """Evaluation and plotting settings."""
    plot_dir: str = "./plots"
    results_csv: str = "./results_summary.csv"
    # CIFAR-100-C corruptions to evaluate
    corruption_types: List[str] = field(
        default_factory=lambda: ["fog", "gaussian_noise", "motion_blur", "brightness"]
    )
    corruption_severities: List[int] = field(default_factory=lambda: [1, 3, 5])


# Convenience singletons
DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
CALIB_CFG = CalibrationConfig()
INFER_CFG = InferenceConfig()
EVAL_CFG = EvalConfig()
