# 4D Radar Adversarial Attack Tool

This tool provides adversarial attack implementations for 4D radar-based 3D object detection models in OpenPCDet.

## Overview

The attack scripts implement FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) attacks on PointPillar models using 4D radar data. The attacks can be applied at the voxel level, targeting different feature dimensions (xyz, doppler, intensity, or all features).

## Files

| File | Attack Type | Target Data | Description |
|------|-------------|-------------|-------------|
| `fgsm_attack.py` | FGSM | Point cloud | Attacks raw point cloud data (point-level) |
| `fgsm_attack_radar.py` | FGSM / PGD | Voxel | Attacks voxelized 4D radar data (voxel-level) |

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

### FGSM Attack

```bash
python attacks/fgsm_attack_radar.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../output/kitti_models/pointpillar_radar/default/ckpt/checkpoint_epoch_80.pth \
    --attack_type fgsm \
    --attack_feature xyz \
    --epsilon 0.05 \
    --num_samples 1296
```

### PGD Attack

```bash
python attacks/fgsm_attack_radar.py \
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
python attacks/fgsm_attack_radar.py \
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
| `--attack_type` | str | fgsm | Attack type: `fgsm` or `pgd` |
| `--attack_feature` | str | all | Target feature: `xyz`, `doppler`, `intensity`, or `all` |
| `--epsilon` | float | 0.05 | Perturbation magnitude |
| `--pgd_steps` | int | 5 | Number of PGD iterations |
| `--num_samples` | int | None | Number of samples to attack (None = all) |
| `--batch_size` | int | 1 | Batch size |
| `--workers` | int | 4 | Number of DataLoader workers (set to 0 if encountering segmentation faults) |

## Output Metrics

| Metric | Definition |
|--------|------------|
| `Original Recall@0.5` | Recall before attack (IoU threshold = 0.5) |
| `Attacked Recall@0.5` | Recall after attack |
| `Attack Success Rate` | Number of successfully attacked samples / Number of originally detected samples |
| `Recall Drop` | Original Recall - Attacked Recall |

**Important**: The Attack Success Rate is calculated only on samples where the model originally detected at least one target with confidence >= 0.5.

## Recommended Configurations

### Quick Test
```bash
python attacks/fgsm_attack_radar.py --attack_type fgsm --epsilon 0.05 --num_samples 100 --workers 0
```

### Standard Evaluation
```bash
python attacks/fgsm_attack_radar.py --attack_type pgd --pgd_steps 10 --epsilon 0.05 --num_samples 1296 --workers 4
```

### High-Strength Attack
```bash
python attacks/fgsm_attack_radar.py --attack_type pgd --pgd_steps 20 --epsilon 0.1 --num_samples 1296
```

## Common Issues

### Segmentation Fault
If you encounter a segmentation fault with `workers > 0`, try:
```bash
python attacks/fgsm_attack_radar.py --workers 0
```

### ModuleNotFoundError: _init_path
Make sure you run the script from the `tools/` directory, not from the project root.

### Different Original Recall Across Runs
This is caused by model state pollution during attack. The script now properly saves and restores model state.

## Output Files

Results are saved to:
```
output/<exp_group>/<tag>/<extra_tag>/attack_results.txt
```

## References

- FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
- PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"