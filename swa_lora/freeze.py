import torch.nn as nn


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)
