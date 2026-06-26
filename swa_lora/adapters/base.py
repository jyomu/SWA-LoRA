from typing import Any, Protocol

import torch.nn as nn

from swa_lora.policy import AttentionPolicy


class ModelAdapter(Protocol):
    """Absorbs model-specific structure so Trainer/loss code stays architecture-agnostic."""

    def base_model(self, model: nn.Module) -> nn.Module: ...

    def decoder_layers(self, model: nn.Module) -> nn.ModuleList: ...

    def self_attention(self, layer: nn.Module) -> nn.Module: ...

    def final_norm(self, model: nn.Module) -> nn.Module: ...

    def lm_head(self, model: nn.Module) -> nn.Module: ...

    def embed_tokens(self, model: nn.Module) -> nn.Module: ...

    def apply_layer_types(self, config: Any, policy: AttentionPolicy) -> None: ...

    def set_sliding_window(self, model: nn.Module, window: int) -> None: ...

    def lora_target_modules(self, include_mlp: bool = False) -> list[str]: ...

    def final_hidden_state(self, model: nn.Module, **forward_kwargs) -> Any: ...

    def layerwise_attention_loss(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        swa_layer_indices: list[int],
        input_ids: Any,
        attention_mask: Any = None,
        lambda_cos: float = 1.0,
    ) -> tuple[Any, dict[int, float]]: ...
