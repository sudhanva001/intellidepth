"""IntelliDepthNet: ResNet backbone with lightweight early-exit heads.

The backbone is a CIFAR-style ResNet (3x3 stem, no maxpool, 3 stages).
Exit heads are attached after configurable stages and produce logits
independently.  During adaptive inference we can stop after any exit
without computing deeper layers.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CFG


class BasicBlock(nn.Module):
    """Standard ResNet basic block for CIFAR.

    Two 3x3 convolutions with batch norm and a shortcut connection.
    When stride != 1 or in_channels != out_channels, the shortcut
    is a 1x1 conv to match dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ExitHead(nn.Module):
    """Stronger classification head with 2 conv layers for better early-exit accuracy.

    Architecture:
        Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> ReLU -> GAP -> FC -> logits
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class IntelliDepthNet(nn.Module):
    """Multi-exit ResNet for CIFAR.

    Args:
        n_blocks_per_stage: Number of residual blocks per stage (e.g. 9 for ResNet-56).
        base_channels: Number of filters in stage 1.
        num_classes: Number of output classes.
        exit_stage_indices: List of stage indices after which to attach exit heads.
            Stage 1 = after first residual stage, etc.  The final classifier
            is always present and is NOT included in this list.
        head_hidden_channels: Hidden dim for exit-head conv layers.
    """

    def __init__(
        self,
        n_blocks_per_stage: int = MODEL_CFG.n_blocks_per_stage,
        base_channels: int = MODEL_CFG.base_channels,
        num_classes: int = MODEL_CFG.num_classes,
        exit_stage_indices: Optional[List[int]] = None,
        head_hidden_channels: int = MODEL_CFG.head_hidden_channels,
    ):
        super().__init__()
        if exit_stage_indices is None:
            exit_stage_indices = MODEL_CFG.exit_stage_indices

        self.exit_stage_indices = sorted(set(exit_stage_indices))
        self.num_exits = len(self.exit_stage_indices) + 1  # +1 for final
        self.num_classes = num_classes

        # --- Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # --- Stages ---
        # Stage 1: base_channels -> base_channels, stride 1
        self.stage1 = self._make_stage(base_channels, base_channels, n_blocks_per_stage, stride=1)
        # Stage 2: base_channels -> base_channels*2, stride 2
        self.stage2 = self._make_stage(base_channels, base_channels * 2, n_blocks_per_stage, stride=2)
        # Stage 3: base_channels*2 -> base_channels*4, stride 2
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, n_blocks_per_stage, stride=2)

        # --- Exit heads ---
        # Map stage index -> channel count
        stage_channels = {
            1: base_channels,
            2: base_channels * 2,
            3: base_channels * 4,
        }
        self.exit_heads = nn.ModuleDict()
        for stage_idx in self.exit_stage_indices:
            if stage_idx not in stage_channels:
                raise ValueError(f"Exit stage index {stage_idx} not in {{1,2,3}}")
            self.exit_heads[str(stage_idx)] = ExitHead(
                stage_channels[stage_idx], num_classes, head_hidden_channels
            )

        # --- Final classifier ---
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_fc = nn.Linear(base_channels * 4, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Full forward pass returning logits from ALL exits.

        Returns:
            List of logits tensors, one per exit (intermediate + final).
        """
        out = self.stem(x)

        logits_list: List[torch.Tensor] = []

        # Stage 1
        out = self.stage1(out)
        if 1 in self.exit_stage_indices:
            logits_list.append(self.exit_heads["1"](out))

        # Stage 2
        out = self.stage2(out)
        if 2 in self.exit_stage_indices:
            logits_list.append(self.exit_heads["2"](out))

        # Stage 3
        out = self.stage3(out)
        if 3 in self.exit_stage_indices:
            logits_list.append(self.exit_heads["3"](out))

        # Final exit
        pooled = self.final_pool(out)
        pooled = pooled.view(pooled.size(0), -1)
        logits_list.append(self.final_fc(pooled))

        return logits_list

    def forward_single_exit(self, x: torch.Tensor, exit_idx: int) -> torch.Tensor:
        """Compute only up to a specific exit (inclusive).

        This is the key method for real latency savings during adaptive
        inference: deeper layers are skipped entirely.

        Args:
            x: Input tensor (B, 3, 32, 32).
            exit_idx: 0-based index into the full exit list.
                0..num_exits-1 correspond to the exits returned by forward().

        Returns:
            Logits tensor for the requested exit.
        """
        out = self.stem(x)

        current_exit = 0

        out = self.stage1(out)
        if 1 in self.exit_stage_indices:
            if current_exit == exit_idx:
                return self.exit_heads["1"](out)
            current_exit += 1

        out = self.stage2(out)
        if 2 in self.exit_stage_indices:
            if current_exit == exit_idx:
                return self.exit_heads["2"](out)
            current_exit += 1

        out = self.stage3(out)
        if 3 in self.exit_stage_indices:
            if current_exit == exit_idx:
                return self.exit_heads["3"](out)
            current_exit += 1

        # Final exit
        if current_exit == exit_idx:
            pooled = self.final_pool(out)
            pooled = pooled.view(pooled.size(0), -1)
            return self.final_fc(pooled)

        raise ValueError(f"exit_idx {exit_idx} out of range for this model")

    def get_exit_flops_ratio(self, exit_idx: int) -> float:
        """Return approximate relative compute cost up to a given exit.

        We approximate FLOPs as proportional to the number of layers executed.
        This is sufficient for accuracy-vs-compute tradeoff curves because
        only relative comparisons matter.
        """
        total_stages = 3  # excluding final pool/fc
        # Each stage has n_blocks_per_stage blocks, each block ~2 conv layers
        # Stem = 1 layer equivalent
        n = MODEL_CFG.n_blocks_per_stage
        stage_costs = {
            1: 1 + n * 2,          # stem + stage1
            2: 1 + n * 2 + n * 2,  # + stage2
            3: 1 + n * 2 + n * 2 + n * 2,  # + stage3
        }
        total_cost = stage_costs[3] + 1  # + final pool/fc

        # Map exit_idx to cumulative cost
        current_exit = 0
        if 1 in self.exit_stage_indices:
            if current_exit == exit_idx:
                return stage_costs[1] / total_cost
            current_exit += 1
        if 2 in self.exit_stage_indices:
            if current_exit == exit_idx:
                return stage_costs[2] / total_cost
            current_exit += 1
        if 3 in self.exit_stage_indices:
            if current_exit == exit_idx:
                return stage_costs[3] / total_cost
            current_exit += 1
        if current_exit == exit_idx:
            return 1.0
        raise ValueError(f"exit_idx {exit_idx} out of range")


if __name__ == "__main__":
    # Sanity check
    model = IntelliDepthNet()
    x = torch.randn(4, 3, 32, 32)
    all_logits = model(x)
    print(f"Number of exits: {len(all_logits)}")
    for i, logits in enumerate(all_logits):
        print(f"  Exit {i}: {logits.shape}")

    for i in range(len(all_logits)):
        single = model.forward_single_exit(x, i)
        assert single.shape == all_logits[i].shape
        print(f"  forward_single_exit({i}) OK, shape {single.shape}")

    print("Model sanity check passed.")
