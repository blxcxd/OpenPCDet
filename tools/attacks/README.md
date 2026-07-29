# 4D Radar Adversarial Attack Tool

This tool provides adversarial attack implementations for 4D radar-based 3D object detection models in OpenPCDet.

## Overview

The attack scripts implement FGSM (Fast Gradient Sign Method) and PGD
(Projected Gradient Descent) attacks on PointPillars models using 4D radar
data. Attacks can optimize either the raw points or the already voxelized
tensor.

## Files

| File | Attack Type | Target Data | Description |
|------|-------------|-------------|-------------|
| `fgsm_attack.py` | FGSM | Generic point cloud | Original non-radar point attack script |
| `../radar_attack/run_attack.py` | FGSM / PGD | 4D radar | Canonical experiment entry point |
| `../radar_attack/attacks/` | FGSM / PGD | Raw 4D radar points | Reusable attacks with a unified output object |
| `../radar_attack/adapters/` | - | OpenPCDet | Differentiable hard-voxelization adapter |
| `../radar_attack/evaluation/` | - | Raw 4D radar points | Metrics and adversarial point-cloud persistence |
| `fgsm_attack_radar.py` | FGSM / PGD | 4D radar | Backward-compatible legacy CLI |
| `radar_point_attack.py` | FGSM / PGD | 4D radar points | Backward-compatible import shim |

## Attack Features

4D radar data consists of 7 features. The attack can target specific feature dimensions:

| Feature | Index | Description |
|---------|-------|-------------|
| `x` | 0 | X coordinate |
| `y` | 1 | Y coordinate |
| `z` | 2 | Z coordinate |
| `rcs` | 3 | Radar cross section (intensity) |
| `v_r` | 4 | Radial velocity (doppler) |
| `v_r_comp` | 5 | Radial velocity component |
| `time` | 6 | Timestamp |

## Usage

### Prerequisites

1. Activate the OpenPCDet environment:
```bash
conda activate openpcdet
```

2. Navigate to the `tools/` directory:
```bash
cd /path/to/OpenPCDet/tools
```

### Point-level FGSM (fixed pillar membership)

This is the recommended first experiment. Every valid point is optimized
independently, padding is never attacked, and xyz coordinates are projected
back into their original pillar.

```bash
python radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type fgsm \
    --attack_feature xyz \
    --epsilon 0.05 \
    --voxel_mode fixed \
    --num_samples 100
```

### Point-level PGD with per-feature budgets

```bash
python radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type pgd \
    --attack_feature all \
    --epsilon 0.05 \
    --epsilon_xyz 0.10 \
    --epsilon_rcs 1.0 \
    --epsilon_doppler 0.20 \
    --epsilon_time 0.01 \
    --pgd_steps 10 \
    --random_start \
    --voxel_mode fixed \
    --save_adv
```

### Point-level PGD allowing points to cross pillars

```bash
python radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type pgd \
    --attack_feature xyz \
    --epsilon 0.20 \
    --pgd_steps 10 \
    --voxel_mode revoxelize
```

Hard voxel assignment is discrete. In `revoxelize` mode, the script rebuilds
the true hard voxels before every forward pass and uses a BPDA/straight-through
gradient through the point-feature gather. In `fixed` mode, the original
assignment is reused and coordinates cannot cross a voxel boundary.

### Voxel-level FGSM

```bash
python radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_domain voxel \
    --attack_type fgsm \
    --attack_feature xyz \
    --epsilon 0.05 \
    --num_samples 1296
```

### Voxel-level PGD

```bash
python radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_type pgd \
    --pgd_steps 10 \
    --attack_feature doppler \
    --epsilon 0.05 \
    --num_samples 1296
```

### Attack All Features

```bash
python radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_type pgd \
    --pgd_steps 10 \
    --attack_feature all \
    --epsilon 0.05
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cfg_file` | str | - | Path to model configuration file |
| `--ckpt` | str | - | Path to trained model checkpoint |
| `--attack_domain` | str | voxel | Radar attack domain: `voxel` or `point` |
| `--attack_type` | str | fgsm | Attack type: `fgsm` or `pgd` |
| `--attack_feature` | str | all | Target feature: `xyz`, `doppler`, `intensity`, or `all` |
| `--epsilon` | float | 0.05 | Perturbation magnitude |
| `--epsilon_xyz` | float | None | Point attack xyz budget (overrides epsilon) |
| `--epsilon_rcs` | float | None | Point attack RCS budget (overrides epsilon) |
| `--epsilon_doppler` | float | None | Point attack velocity budget (overrides epsilon) |
| `--epsilon_time` | float | None | Point attack time budget (overrides epsilon) |
| `--pgd_steps` | int | 10 point / 5 voxel | Number of PGD iterations |
| `--voxel_mode` | str | fixed | Point attack topology: `fixed` or `revoxelize` |
| `--random_start` | flag | off | Random PGD initialization inside the budget |
| `--num_samples` | int | None | Number of samples to attack (None = all) |
| `--score_threshold` | float | 0.5 | Confidence threshold for sample-level success |
| `--save_adv` | flag | off | Save adversarial raw point clouds; requires point domain |
| `--adv_format` | str | npy | Saved point-cloud format: `npy` or headerless `bin` |
| `--adv_dir` | str | None | Custom save directory; defaults inside experiment output |
| `--batch_size` | int | 1 | Batch size |
| `--workers` | int | 4 | Number of DataLoader workers (set to 0 if encountering segmentation faults) |

## Output Metrics

| Metric | Definition |
|--------|------------|
| `Original Recall@0.5` | Recall before attack (IoU threshold = 0.5) |
| `Attacked Recall@0.5` | Recall after attack |
| `Attack Success Rate` | Number of successfully attacked samples / Number of originally detected samples |
| `Recall Drop` | Original Recall - Attacked Recall |
| `Max / Mean \|delta\|` | Raw point-feature perturbation statistics |

**Important**: The Attack Success Rate is calculated only on samples where the model originally detected at least one target with confidence >= 0.5.

## Recommended Configurations

### Quick Test
```bash
python radar_attack/run_attack.py --attack_type fgsm --epsilon 0.05 --num_samples 100 --workers 0
```

### Standard Evaluation
```bash
python radar_attack/run_attack.py --attack_type pgd --pgd_steps 10 --epsilon 0.05 --num_samples 1296 --workers 4
```

### High-Strength Attack
```bash
python radar_attack/run_attack.py --attack_type pgd --pgd_steps 20 --epsilon 0.1 --num_samples 1296
```

## Common Issues

### Segmentation Fault
If you encounter a segmentation fault with `workers > 0`, try:
```bash
python radar_attack/run_attack.py --workers 0
```

### ModuleNotFoundError: _init_path
Make sure you run the script from the `tools/` directory, not from the project root.

### Different Original Recall Across Runs
This is caused by model state pollution during attack. The script now properly saves and restores model state.

### Raw points have no gradient

Standard hard-voxelized PointPillars consumes `batch_dict['voxels']`, not
`batch_dict['points']`. Simply setting raw points to `requires_grad=True`
therefore does not work. The `--attack_domain point` branch solves this by
rebuilding voxels from the raw point tensor while keeping the gathered point
features connected to autograd.
BatchNorm statistics are frozen during loss computation.

## Output Files

Results are saved to:
```
output/<exp_group>/<tag>/<extra_tag>/attack_results.txt
```

With `--save_adv`, every sample is additionally saved as a raw feature array
without OpenPCDet's leading batch-index column:

```
output/<exp_group>/<tag>/<extra_tag>/adversarial_points/<frame_id>.npy
output/<exp_group>/<tag>/<extra_tag>/adversarial_points/manifest.jsonl
```

The manifest records feature order, tensor shape, dtype, attack settings, and
per-frame perturbation statistics. A `.bin` file is headerless float32 data;
use its manifest `shape` field when loading it. These are adversarial point
clouds, not voxel tensors, so they can be passed through another detector's
own preprocessing for transfer-attack evaluation.

## References

- FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
- PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
