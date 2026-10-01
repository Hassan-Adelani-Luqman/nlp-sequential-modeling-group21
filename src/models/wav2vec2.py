"""A5: a self-supervised pretrained speech Transformer (wav2vec 2.0 / XLS-R), fine-tuned for the 12 words.

* Input: the raw 16 kHz waveform of the centred energy window, shape (batch, samples). Each clip is
  normalised to zero mean and unit variance inside the model, as the models' own feature extractor does.
* Network: Hugging Face ``Wav2Vec2ForSequenceClassification``. A convolutional feature encoder turns the
  waveform into one vector per 20 ms, a Transformer contextualises them with self-attention, and the
  frames are projected to 256 dimensions, mean-pooled over time and classified.
* Backbones: ``"xlsr-300m"`` (XLS-R, 300M parameters, pretrained on 128 languages including Swahili) and
  ``"wav2vec2-base"`` (95M parameters, pretrained on English audiobooks only). Comparing them tests
  whether multilingual pretraining transfers better to Swahili.
* Fine-tuning: ``freeze="feature_encoder"`` (default) trains the Transformer and the head, as is standard;
  ``"encoder"`` trains only the head on frozen representations. The new, randomly initialised head
  learns at its own higher rate (``head_lr``), via ``optimizer_groups`` in ``src.train_torch``.
* Both backbones are fine-tuned with the same regularisation (time masking of 10-frame spans with
  probability ``mask_time_prob``, no LayerDrop), overriding their different pretraining defaults.

Usage::

    model = Wav2Vec2Classifier("xlsr-300m")                    # downloads the pretrained weights
    logits = model(torch.randn(4, 24000))                      # (4, 12)
    Wav2Vec2Classifier("xlsr-300m", pretrained=False)          # architecture only: to load a checkpoint into
"""
from __future__ import annotations

import torch
from torch import nn

MODEL_IDS = {"xlsr-300m": "facebook/wav2vec2-xls-r-300m", "wav2vec2-base": "facebook/wav2vec2-base"}
FREEZE = ("feature_encoder", "encoder")


def tiny_config(**overrides):
    """A few-thousand-parameter wav2vec 2.0 architecture with random weights, for tests and smoke runs."""
    from transformers import Wav2Vec2Config

    kw = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
              conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
              classifier_proj_size=8)
    return Wav2Vec2Config(**{**kw, **overrides})


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, backbone: str = "xlsr-300m", n_classes: int = 12, freeze: str = "feature_encoder",
                 head_lr: float = 1e-3, weighted_layer_sum: bool = False, mask_time_prob: float = 0.05,
                 layerdrop: float = 0.0, pretrained: bool = True, config=None):
        super().__init__()
        from transformers import AutoConfig, AutoModelForAudioClassification

        if freeze not in FREEZE:
            raise ValueError(f"freeze must be one of {FREEZE}")
        self.head_lr = head_lr
        overrides = dict(num_labels=n_classes, use_weighted_layer_sum=weighted_layer_sum,
                         mask_time_prob=mask_time_prob, layerdrop=layerdrop)
        if config is None:
            name = MODEL_IDS.get(backbone, backbone)
            config = AutoConfig.from_pretrained(name, **overrides)
        else:
            name, pretrained = None, False
            base = {k: v for k, v in config.to_dict().items() if k not in ("id2label", "label2id")}
            config = type(config).from_dict({**base, **overrides})
        # Eager attention: its backward pass is deterministic, and for ~75 frames it costs little.
        if pretrained:
            self.net = AutoModelForAudioClassification.from_pretrained(name, config=config, attn_implementation="eager")
        else:
            self.net = AutoModelForAudioClassification.from_config(config, attn_implementation="eager")
        self.net.float()
        if weighted_layer_sum:
            # The layer weights are not in the pretrained checkpoint, and from_pretrained leaves this plain
            # Parameter uninitialised (arbitrary memory). Set the intended start: equal weight on every layer.
            with torch.no_grad():
                self.net.layer_weights.fill_(1.0 / self.net.layer_weights.numel())
        self.net.freeze_feature_encoder()
        if freeze == "encoder":
            self.net.freeze_base_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True) + 1e-7)
        return self.net(input_values=x).logits

    def optimizer_groups(self, lr: float) -> list[dict]:
        """Trainable parameters: the pretrained encoder at ``lr``, the new head at ``head_lr``."""
        prefix = self.net.base_model_prefix + "."
        trainable = [(n, p) for n, p in self.net.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in trainable if n.startswith(prefix)], "lr": lr},
                  {"params": [p for n, p in trainable if not n.startswith(prefix)], "lr": self.head_lr}]
        return [g for g in groups if g["params"]]

    def layer_weights(self) -> list[float] | None:
        """Softmax weights over the input embedding and each Transformer layer (``weighted_layer_sum`` only)."""
        if not self.net.config.use_weighted_layer_sum:
            return None
        return torch.softmax(self.net.layer_weights.detach().float(), 0).cpu().tolist()


def count_all_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)
