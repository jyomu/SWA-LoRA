import copy
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from swa_lora.adapters.qwen3 import Qwen3Adapter
from swa_lora.policy import AttentionPolicy


@dataclass
class PretrainedSetup:
    teacher: PreTrainedModel
    student: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    adapter: Qwen3Adapter
    policy: AttentionPolicy


def build_pretrained_setup(
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    num_full_top_layers: int = 1,
    sliding_window: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    attn_implementation: str = "sdpa",
) -> PretrainedSetup:
    """Load the pretrained Full-Attention model twice: once as a frozen teacher,
    once as a student whose lower layers are reconfigured to sliding-window
    attention. Both start from identical weights -- only config.layer_types
    (and thus the causal mask each layer sees) differs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation=attn_implementation
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    adapter = Qwen3Adapter()
    num_hidden_layers = teacher.config.num_hidden_layers
    policy = AttentionPolicy.top_full_layers(num_hidden_layers, num_full_top_layers, sliding_window)

    student_config = copy.deepcopy(teacher.config)
    adapter.apply_layer_types(student_config, policy)

    student = AutoModelForCausalLM.from_pretrained(
        model_name, config=student_config, dtype=dtype, attn_implementation=attn_implementation
    ).to(device)

    return PretrainedSetup(teacher=teacher, student=student, tokenizer=tokenizer, adapter=adapter, policy=policy)
