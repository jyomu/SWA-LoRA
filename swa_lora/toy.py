import copy
from dataclasses import dataclass

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from swa_lora.adapters.qwen3 import Qwen3Adapter
from swa_lora.policy import AttentionPolicy


@dataclass
class ToySetup:
    teacher: Qwen3ForCausalLM
    student: Qwen3ForCausalLM
    adapter: Qwen3Adapter
    policy: AttentionPolicy
    config: Qwen3Config


def build_toy_setup(
    num_hidden_layers: int = 4,
    num_full_top_layers: int = 1,
    sliding_window: int = 2,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    intermediate_size: int = 64,
    vocab_size: int = 97,
    attn_implementation: str = "eager",
    seed: int = 0,
) -> ToySetup:
    """Phase 0 toy model: tiny random-init Qwen3, teacher = all-full-attention,
    student = same init weights with lower layers switched to SWA."""
    torch.manual_seed(seed)
    teacher_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=hidden_size // num_attention_heads,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attn_implementation=attn_implementation,
    )

    teacher = Qwen3ForCausalLM(teacher_config)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    adapter = Qwen3Adapter()
    policy = AttentionPolicy.top_full_layers(num_hidden_layers, num_full_top_layers, sliding_window)

    student_config = copy.deepcopy(teacher_config)
    adapter.apply_layer_types(student_config, policy)
    student = Qwen3ForCausalLM(student_config)
    student.load_state_dict(teacher.state_dict())

    return ToySetup(teacher=teacher, student=student, adapter=adapter, policy=policy, config=teacher_config)
