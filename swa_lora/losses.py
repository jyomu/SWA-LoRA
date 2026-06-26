import torch
import torch.nn.functional as F


def hidden_state_loss(
    student_z: torch.Tensor, teacher_z: torch.Tensor, lambda_cos: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L_hidden = MSE(stopgrad(z_T), z_S) + lambda_cos * (1 - cosine(z_T, z_S))."""
    teacher_z = teacher_z.detach()
    mse = F.mse_loss(student_z, teacher_z)
    cos_sim = F.cosine_similarity(student_z, teacher_z, dim=-1).mean()
    cos_loss = 1.0 - cos_sim
    total = mse + lambda_cos * cos_loss
    return total, {"mse": mse.detach(), "cos": cos_loss.detach()}
