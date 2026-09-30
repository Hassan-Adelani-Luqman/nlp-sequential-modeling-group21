"""A3: recurrent sequence model - (Bi)LSTM over log-mel frames with additive attention pooling.

Architecture (defaults follow Plan.md, Phase 5)::

    log-mel (B, T, 64) -> [optional Conv1d front-end] -> 2-layer BiLSTM (128 per direction, dropout 0.3)
                       -> attention pooling over time -> dropout -> Linear(256 -> 12)

The recurrence reads the frames in order and carries information forward (and
backward, when bidirectional) through its hidden state (Hochreiter & Schmidhuber,
1997; Graves et al., 2013). Attention pooling uses Bahdanau et al.'s (2015)
additive scoring to learn *which* frames matter, instead of relying only on the
last hidden state. The per-frame attention weights are returned for the
error-analysis plots, e.g. whether the model attends to the vowel that separates
*nne* from *nane*.

de Andrade et al. (2018) put convolutional layers *before* the recurrence and
attention for speech commands. Our default leaves them out, so that A3 stays a
purely recurrent model and contrasts cleanly with the convolutional A4.
``conv_frontend=True`` restores their design for the R2 comparison.

Ablation switches (R2): ``bidirectional`` (a unidirectional model can stream),
``pooling`` ("attention" | "last" | "mean") and ``conv_frontend``.
"""
from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """score_t = v^T tanh(W h_t); weights = softmax over time; output = sum_t weights_t h_t."""

    def __init__(self, dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(torch.tanh(self.proj(h))).squeeze(-1), dim=1)  # (B, T)
        return torch.bmm(weights.unsqueeze(1), h).squeeze(1), weights


class BiLSTMAttention(nn.Module):
    def __init__(self, n_features: int = 64, n_classes: int = 12, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True, pooling: str = "attention",
                 conv_frontend: bool = False):
        super().__init__()
        if pooling not in ("attention", "last", "mean"):
            raise ValueError("pooling must be 'attention', 'last' or 'mean'")
        self.pooling, self.bidirectional = pooling, bidirectional
        self.frontend = None
        rnn_in = n_features
        if conv_frontend:  # local spectral-temporal patterns before the recurrence
            self.frontend = nn.Sequential(
                nn.Conv1d(n_features, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU())
            rnn_in = 128
        self.lstm = nn.LSTM(rnn_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.attention = AdditiveAttention(out_dim) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """``x``: (batch, frames, features). Returns logits, plus (batch, frames) attention weights if asked."""
        if self.frontend is not None:
            x = self.frontend(x.transpose(1, 2)).transpose(1, 2)
        h, (h_n, _) = self.lstm(x)
        weights = None
        if self.pooling == "attention":
            pooled, weights = self.attention(h)
        elif self.pooling == "last":
            # final state of the forward direction (+ final state of the backward direction)
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        else:
            pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return (logits, weights) if return_attention else logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
