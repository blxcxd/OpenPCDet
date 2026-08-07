from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class AttackOutput:
    """Model-independent output returned by every raw-point attack."""

    adv_points: torch.Tensor
    model_inputs: Dict[str, torch.Tensor]
    stats: Dict[str, float]


def set_attack_mode(model: nn.Module) -> Dict[nn.Module, bool]:
    """Enable detector losses while freezing BatchNorm running statistics."""
    states = {module: module.training for module in model.modules()}
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return states


def restore_attack_modes(states: Dict[nn.Module, bool]) -> None:
    """Restore the exact train/eval state of every model module."""
    for module, training in states.items():
        module.training = training
