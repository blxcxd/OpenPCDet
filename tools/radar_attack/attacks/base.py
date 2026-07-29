from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class AttackOutput:
    """Model-independent output returned by every raw-point attack."""

    adv_points: torch.Tensor
    model_inputs: Dict[str, torch.Tensor]
    stats: Dict[str, float]
