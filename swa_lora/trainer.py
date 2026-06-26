import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from swa_lora.adapters.base import ModelAdapter
from swa_lora.losses import hidden_state_loss


@dataclass
class TrainerConfig:
    grad_accum_steps: int = 1
    lambda_cos: float = 1.0
    lambda_ce: float = 0.0
    lambda_local: float = 0.0  # plan section 6.3 per-layer local cutoff loss; 0 disables
    max_grad_norm: float | None = 1.0
    amp_dtype: torch.dtype | None = None


class Trainer:
    """Sequential teacher/student forward, hidden-state distillation loss,
    optional CE regularizer, optional per-layer local loss, gradient
    accumulation, mixed precision."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        adapter: ModelAdapter,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig | None = None,
        swa_layer_indices: list[int] | None = None,
    ):
        self.teacher = teacher
        self.student = student
        self.adapter = adapter
        self.optimizer = optimizer
        self.config = config or TrainerConfig()
        self.swa_layer_indices = swa_layer_indices
        self.teacher.eval()
        self._accum_count = 0

        if self.config.lambda_local > 0 and not swa_layer_indices:
            raise ValueError("swa_layer_indices is required when lambda_local > 0")

    def _autocast(self):
        if self.config.amp_dtype is None:
            return contextlib.nullcontext()
        device_type = next(self.student.parameters()).device.type
        return torch.autocast(device_type=device_type, dtype=self.config.amp_dtype)

    @torch.no_grad()
    def _teacher_forward(self, **batch) -> torch.Tensor:
        return self.adapter.final_hidden_state(self.teacher, **batch)

    def _student_forward(self, **batch) -> torch.Tensor:
        return self.adapter.final_hidden_state(self.student, **batch)

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | bool]:
        with self._autocast():
            teacher_z = self._teacher_forward(input_ids=input_ids, attention_mask=attention_mask)
            student_z = self._student_forward(input_ids=input_ids, attention_mask=attention_mask)
            loss, metrics = hidden_state_loss(student_z, teacher_z, lambda_cos=self.config.lambda_cos)

            if self.config.lambda_ce > 0:
                if labels is None:
                    raise ValueError("labels are required when lambda_ce > 0")
                logits = self.adapter.lm_head(self.student)(student_z)
                ce = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + self.config.lambda_ce * ce
                metrics["ce"] = ce.detach()

            if self.config.lambda_local > 0:
                local, local_per_layer = self.adapter.layerwise_attention_loss(
                    self.teacher,
                    self.student,
                    self.swa_layer_indices,
                    input_ids,
                    attention_mask=attention_mask,
                    lambda_cos=self.config.lambda_cos,
                )
                loss = loss + self.config.lambda_local * local
                metrics["local"] = local.detach()
                for layer_idx, layer_val in local_per_layer.items():
                    metrics[f"local_layer_{layer_idx}"] = layer_val

        (loss / self.config.grad_accum_steps).backward()
        self._accum_count += 1

        stepped = False
        if self._accum_count >= self.config.grad_accum_steps:
            if self.config.max_grad_norm is not None:
                trainable = (p for p in self.student.parameters() if p.requires_grad)
                torch.nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
            stepped = True

        metrics["loss"] = loss.detach()
        metrics["stepped"] = stepped
        return metrics

    @torch.no_grad()
    def evaluate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        was_training = self.student.training
        self.student.eval()
        teacher_z = self._teacher_forward(input_ids=input_ids, attention_mask=attention_mask)
        student_z = self._student_forward(input_ids=input_ids, attention_mask=attention_mask)
        _, metrics = hidden_state_loss(student_z, teacher_z, lambda_cos=self.config.lambda_cos)
        if was_training:
            self.student.train()
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.student, "save_pretrained"):
            self.student.save_pretrained(path)
        else:
            torch.save(self.student.state_dict(), path / "student.pt")
