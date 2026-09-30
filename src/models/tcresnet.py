"""A4: convolutional sequence model - TC-ResNet (Choi et al., 2019).

The frequency axis is treated as channels, and 1-D convolutions slide **over time only**.
Each layer therefore mixes information from neighbouring frames, and stacking strided
residual blocks widens the temporal context step by step::

    features (B, T, F) -> Conv1d(F -> c0, k=3) -> residual blocks (k=9, stride 2) -> mean over time -> Linear(-> 12)

TC-ResNet8 has 3 residual blocks with channels (16, 24, 32, 48) x ``width``; Choi et al.'s
strongest small model uses ``width=1.5``. TC-ResNet14 (``blocks=6``) doubles the depth.

Because the classifier follows global average pooling, applying its weights to every time
step before pooling gives an exact class activation map over time (``return_cam=True``):
how strongly each part of the clip supports each word. Unlike the recurrent A3, context
comes only from the convolutional receptive field (``receptive_field_frames``).
"""
from __future__ import annotations

import torch
from torch import nn

CHANNELS = {3: (16, 24, 32, 48), 6: (16, 24, 24, 32, 32, 48, 48)}  # TC-ResNet8 / TC-ResNet14


class ResidualBlock1d(nn.Module):
    """Two k-wide temporal convolutions (the first strided) with a projected shortcut when shapes change."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 9, stride: int = 2):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel, stride=1, padding=pad, bias=False), nn.BatchNorm1d(c_out))
        self.shortcut = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out), nn.ReLU())
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class TCResNet(nn.Module):
    def __init__(self, n_features: int = 64, n_classes: int = 12, blocks: int = 3, width: float = 1.5,
                 kernel: int = 9, dropout: float = 0.5):
        super().__init__()
        if blocks not in CHANNELS:
            raise ValueError(f"blocks must be one of {sorted(CHANNELS)}")
        ch = [int(round(c * width)) for c in CHANNELS[blocks]]
        # TC-ResNet14 keeps stride 2 only in the first block of each channel stage
        strides = [2] * blocks if blocks == 3 else [2, 1, 2, 1, 2, 1]
        self.kernel, self.strides = kernel, strides
        self.stem = nn.Conv1d(n_features, ch[0], kernel_size=3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ResidualBlock1d(ch[i], ch[i + 1], kernel, strides[i]) for i in range(blocks)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(ch[-1], n_classes)

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        """``x``: (batch, frames, features). With ``return_cam`` also returns (batch, classes, frames') activations."""
        h = self.blocks(self.stem(x.transpose(1, 2)))       # (B, C, T')
        logits = self.head(self.dropout(h.mean(dim=-1)))
        if not return_cam:
            return logits
        cam = torch.einsum("kc,bct->bkt", self.head.weight, h) + self.head.bias[None, :, None]
        return logits, cam                                    # cam averaged over time equals the logits

    def receptive_field_frames(self) -> int:
        """How many input frames one output step can see (10 ms per frame)."""
        rf, jump = 3, 1                                       # the stem: kernel 3, stride 1
        for s in self.strides:
            rf += (self.kernel - 1) * jump
            jump *= s
            rf += (self.kernel - 1) * jump
        return rf
