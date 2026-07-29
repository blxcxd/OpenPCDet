"""FGSM/PGD baselines on OpenPCDet's voxel feature tensor."""

from typing import Dict, Sequence

import torch
import torch.nn as nn


DEFAULT_RADAR_FEATURES = (
    'x',
    'y',
    'z',
    'rcs',
    'v_r',
    'v_r_comp',
    'time',
)


def _feature_group(name: str) -> str:
    name = name.lower()
    if name in {'x', 'y', 'z'}:
        return 'xyz'
    if name in {'rcs', 'intensity', 'power'}:
        return 'rcs'
    if name in {
        'v_r',
        'v_r_comp',
        'velocity',
        'velocity_comp',
        'doppler',
        'doppler_comp',
    }:
        return 'doppler'
    if name in {'time', 'timestamp'}:
        return 'time'
    return 'other'


def build_feature_mask(
    tensor: torch.Tensor,
    attack_feature: str,
    feature_names: Sequence[str] = DEFAULT_RADAR_FEATURES,
) -> torch.Tensor:
    """Return a broadcastable mask selecting semantic radar features."""
    if tensor.shape[-1] != len(feature_names):
        raise ValueError(
            f'voxel tensor has {tensor.shape[-1]} features but '
            f'{len(feature_names)} feature names were provided'
        )

    selected_group = 'rcs' if attack_feature == 'intensity' else attack_feature
    selected = [
        selected_group == 'all' or _feature_group(name) == selected_group
        for name in feature_names
    ]
    if not any(selected):
        raise ValueError(
            f'feature group "{attack_feature}" is absent from '
            f'{list(feature_names)}'
        )
    shape = [1] * tensor.ndim
    shape[-1] = len(selected)
    return tensor.new_tensor(selected).reshape(shape)


def _set_attack_mode(model: nn.Module) -> Dict[nn.Module, bool]:
    states = {module: module.training for module in model.modules()}
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return states


def _restore_modes(states: Dict[nn.Module, bool]) -> None:
    for module, training in states.items():
        module.training = training


def _project_voxels(
    candidate: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    point_cloud_range: Sequence[float],
    feature_names: Sequence[str],
) -> torch.Tensor:
    projected = torch.maximum(
        torch.minimum(candidate, original + epsilon),
        original - epsilon,
    )

    spatial_limits = {
        'x': (point_cloud_range[0], point_cloud_range[3]),
        'y': (point_cloud_range[1], point_cloud_range[4]),
        'z': (point_cloud_range[2], point_cloud_range[5]),
    }
    for column, name in enumerate(feature_names):
        if name.lower() in spatial_limits:
            lower, upper = spatial_limits[name.lower()]
            projected[..., column] = projected[..., column].clamp(lower, upper)
    return projected


def voxel_attack(
    model: nn.Module,
    batch_dict: Dict,
    epsilon: float,
    attack_type: str = 'fgsm',
    attack_feature: str = 'all',
    steps: int = 5,
    point_cloud_range: Sequence[float] = (0, -25.6, -3, 51.2, 25.6, 2),
    feature_names: Sequence[str] = DEFAULT_RADAR_FEATURES,
) -> torch.Tensor:
    """Generate an untargeted adversarial voxel tensor."""
    if attack_type not in {'fgsm', 'pgd'}:
        raise ValueError('attack_type must be "fgsm" or "pgd"')
    if epsilon < 0:
        raise ValueError('epsilon must be non-negative')
    if steps <= 0:
        raise ValueError('steps must be positive')

    original = batch_dict['voxels'].detach().clone()
    feature_mask = build_feature_mask(
        original,
        attack_feature,
        feature_names,
    )
    iterations = 1 if attack_type == 'fgsm' else steps
    step_size = epsilon if attack_type == 'fgsm' else 2.0 * epsilon / steps
    adversarial = original.clone()

    states = _set_attack_mode(model)
    try:
        for _ in range(iterations):
            adversarial = adversarial.detach().requires_grad_(True)
            attack_batch = dict(batch_dict)
            attack_batch['voxels'] = adversarial
            ret_dict, _, _ = model(attack_batch)
            loss = ret_dict['loss'].mean()
            gradient = torch.autograd.grad(loss, adversarial)[0]
            if not torch.isfinite(gradient).all():
                raise RuntimeError('non-finite voxel gradient during radar attack')

            adversarial = _project_voxels(
                adversarial
                + step_size * gradient.sign() * feature_mask,
                original,
                epsilon,
                point_cloud_range,
                feature_names,
            )
    finally:
        _restore_modes(states)
        model.zero_grad(set_to_none=True)

    return adversarial.detach()


def fgsm_attack_voxel(
    model: nn.Module,
    batch_dict: Dict,
    epsilon: float,
    attack_feature: str = 'all',
    point_cloud_range: Sequence[float] = (0, -25.6, -3, 51.2, 25.6, 2),
    feature_names: Sequence[str] = DEFAULT_RADAR_FEATURES,
) -> torch.Tensor:
    return voxel_attack(
        model,
        batch_dict,
        epsilon,
        attack_type='fgsm',
        attack_feature=attack_feature,
        point_cloud_range=point_cloud_range,
        feature_names=feature_names,
    )


def pgd_attack_voxel(
    model: nn.Module,
    batch_dict: Dict,
    epsilon: float,
    attack_feature: str = 'all',
    steps: int = 5,
    point_cloud_range: Sequence[float] = (0, -25.6, -3, 51.2, 25.6, 2),
    feature_names: Sequence[str] = DEFAULT_RADAR_FEATURES,
) -> torch.Tensor:
    return voxel_attack(
        model,
        batch_dict,
        epsilon,
        attack_type='pgd',
        attack_feature=attack_feature,
        steps=steps,
        point_cloud_range=point_cloud_range,
        feature_names=feature_names,
    )
