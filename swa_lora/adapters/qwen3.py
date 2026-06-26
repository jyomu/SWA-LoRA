from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from swa_lora.policy import AttentionPolicy


class Qwen3Adapter:
    """ModelAdapter for Qwen3 (transformers Qwen3ForCausalLM)."""

    def _unwrap_peft(self, model: nn.Module) -> nn.Module:
        # peft's get_peft_model injects LoRA layers in place and exposes the
        # original transformers module tree via get_base_model().
        if hasattr(model, "get_base_model"):
            return model.get_base_model()
        return model

    def base_model(self, model: nn.Module) -> nn.Module:
        """Qwen3Model body (no LM head)."""
        return self._unwrap_peft(model).model

    def decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        return self.base_model(model).layers

    def self_attention(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def final_norm(self, model: nn.Module) -> nn.Module:
        return self.base_model(model).norm

    def lm_head(self, model: nn.Module) -> nn.Module:
        return self._unwrap_peft(model).lm_head

    def embed_tokens(self, model: nn.Module) -> nn.Module:
        return self.base_model(model).embed_tokens

    def apply_layer_types(self, config: Any, policy: AttentionPolicy) -> None:
        config.layer_types = list(policy.layer_types)
        config.sliding_window = policy.sliding_window
        config.use_sliding_window = True

    def set_sliding_window(self, model: nn.Module, window: int) -> None:
        """Change the SWA window mid-training. Qwen3Model rebuilds the
        sliding-window mask from config.sliding_window on every forward call
        (not cached), so mutating the shared config object is enough for the
        mask; each Qwen3Attention also snapshots sliding_window once at
        __init__, so that per-layer attribute needs updating too."""
        body = self.base_model(model)
        body.config.sliding_window = window
        for layer in self.decoder_layers(model):
            attn = self.self_attention(layer)
            if getattr(attn, "layer_type", None) == "sliding_attention":
                attn.sliding_window = window

    def lora_target_modules(self, include_mlp: bool = False) -> list[str]:
        modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if include_mlp:
            modules += ["gate_proj", "up_proj", "down_proj"]
        return modules

    def final_hidden_state(self, model: nn.Module, **forward_kwargs):
        forward_kwargs.setdefault("use_cache", False)
        forward_kwargs.setdefault("output_hidden_states", False)
        body = self.base_model(model)
        return body(**forward_kwargs).last_hidden_state

    def layerwise_attention_loss(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        swa_layer_indices: list[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        lambda_cos: float = 1.0,
    ) -> tuple[torch.Tensor, dict[int, float]]:
        """Plan section 6.3 "local cutoff loss": for each SWA layer, feed the
        *teacher's* (frozen, detached) hidden state into that layer's
        self-attention under both the full-attention mask (teacher's own
        weights) and the SWA mask (student's LoRA-adapted weights), and
        penalize the difference. Because the shared input never requires
        grad, gradients from layer i's term cannot reach any other layer's
        parameters -- each layer's loss only trains its own LoRA.

        Returns the mean loss (for backprop) plus a per-layer breakdown
        (detached floats, keyed by layer index) for logging.
        """
        teacher_body = self.base_model(teacher_model)
        student_body = self.base_model(student_model)
        teacher_layers = self.decoder_layers(teacher_model)
        student_layers = self.decoder_layers(student_model)

        with torch.no_grad():
            inputs_embeds = teacher_body.embed_tokens(input_ids)
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            position_embeddings = teacher_body.rotary_emb(inputs_embeds, position_ids)
            mask_kwargs = {
                "config": teacher_body.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            full_mask = create_causal_mask(**mask_kwargs)
            sliding_mask = create_sliding_window_causal_mask(**{**mask_kwargs, "config": student_body.config})

            teacher_out = teacher_body(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
            )
            hidden_states = teacher_out.hidden_states  # tuple, already detached via no_grad

        losses = []
        per_layer: dict[int, float] = {}
        for i in swa_layer_indices:
            layer_input = hidden_states[i]
            with torch.no_grad():
                # .detach() defends against an unfrozen teacher (e.g. in
                # tests): without it, gradients would leak into the
                # teacher's input_layernorm weight even though it never
                # contributes to the student's loss term.
                normed = teacher_layers[i].input_layernorm(layer_input).detach()

                teacher_attn_out, _ = teacher_layers[i].self_attn(
                    hidden_states=normed,
                    attention_mask=full_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )

            student_attn_out, _ = student_layers[i].self_attn(
                hidden_states=normed,
                attention_mask=sliding_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

            mse = F.mse_loss(student_attn_out, teacher_attn_out)
            cos = 1.0 - F.cosine_similarity(student_attn_out, teacher_attn_out, dim=-1).mean()
            term = mse + lambda_cos * cos
            losses.append(term)
            per_layer[i] = term.detach().item()

        return torch.stack(losses).mean(), per_layer
