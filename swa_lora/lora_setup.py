import torch.nn as nn
from peft import LoraConfig, get_peft_model

from swa_lora.adapters.base import ModelAdapter
from swa_lora.policy import AttentionPolicy


def apply_lora(
    model: nn.Module,
    adapter: ModelAdapter,
    policy: AttentionPolicy,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    include_mlp: bool = False,
):
    """Wrap `model` with LoRA restricted to the SWA layers' projection modules.

    get_peft_model freezes every base parameter first, then injects trainable
    LoRA params only on the (layer, module) pairs matched by target_modules +
    layers_to_transform. Top full-attention layers, final norm, lm_head and
    embeddings are therefore frozen automatically with no extra code.
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=adapter.lora_target_modules(include_mlp=include_mlp),
        layers_to_transform=policy.swa_layer_indices,
        bias="none",
    )
    return get_peft_model(model, lora_config)
