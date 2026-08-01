# 4D Radar Adversarial Attack

`radar_attack` 是面向 OpenPCDet 4D 雷达检测模型的白盒对抗攻击实验模块。目前支持在原始雷达点云和体素特征上运行 FGSM、PGD，并比较攻击前后的检测指标和 View-of-Delft（VoD）官方 AP。

本目录不依赖旧的 `tools/attacks` 实现，但仍依赖 OpenPCDet 的数据集、模型、损失函数和配置系统。

## 目录结构

```text
tools/radar_attack/
├── run_attack.py                 # 推荐的命令行入口
├── runner.py                     # 加载数据和模型，执行攻击、推理与评估
├── attacks/
│   ├── base.py                   # 统一的 AttackOutput
│   ├── gradient.py               # 原始点云 FGSM/PGD
│   └── voxel.py                  # 体素 FGSM/PGD 对照基线
├── adapters/
│   └── openpcdet.py              # 可微分点云到 PointPillars 体素适配
├── evaluation/
│   ├── metrics.py                # 扰动、Recall 和攻击成功率
│   ├── storage.py                # 保存对抗点云与 manifest
│   └── vod.py                    # VoD 官方 AP 评估适配
└── tests/
    └── test_components.py        # 组件测试
```

推荐入口是：

```bash
python tools/radar_attack/run_attack.py ...
```

旧的 `tools/attacks/fgsm_attack_radar.py` 只保留为兼容入口，新实验应使用上面的命令。

## 点云攻击与体素攻击

### 点云攻击

使用 `--attack_domain point` 时，攻击变量是 OpenPCDet batch 中的原始雷达点：

```text
[batch_index, x, y, z, rcs, v_r, v_r_comp, time]
```

攻击产生 `adv_points`，随后通过 `PointCloudVoxelizer` 转换成 PointPillars 的输入。保存文件时会去掉 `batch_index`，得到真正的对抗点云，而不是体素张量。

点云攻击提供两种体素拓扑模式：

- `fixed`：攻击期间保持原始 pillar 分配，空间坐标不会跨越原 pillar 边界。梯度稳定，但限制更强。
- `revoxelize`：每一步根据当前对抗点重新划分 pillar，并使用 BPDA/straight-through 特征聚合近似通过离散体素化反向传播。

### 体素攻击

使用 `--attack_domain voxel` 时，攻击变量是 OpenPCDet 已生成的 `voxels` 张量。这是用于比较的基线，不会产生可直接复用的原始对抗点云，也不支持 `--save_adv`。

## 环境与数据

在 OpenPCDet 仓库根目录运行命令，并使用已经安装 OpenPCDet 的 Python 环境：

```bash
conda activate openpcdet
```

当前 PointPillars 雷达配置为：

```text
tools/cfgs/kitti_models/pointpillar_radar.yaml
```

程序启动后会把工作目录切换到 `tools`，因此 `--cfg_file` 应写成相对于 `tools` 的路径：

```text
cfgs/kitti_models/pointpillar_radar.yaml
```

建议为 `--ckpt` 使用绝对路径。

## 统计雷达特征分布

在选择 epsilon 前，先统计进入模型的验证集雷达点特征：

```bash
python tools/radar_attack/analyze_features.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --batch_size 8 \
    --workers 4
```

默认统计实际被硬体素化保留、进入 PointPillars VFE 的有效点，不包含体素零填充，也不包含因范围、每体素点数上限或体素数量上限而被丢弃的点。该命令不加载 checkpoint，也不需要 GPU。

如需审计经过特征编码、相机 FOV 和 `DATA_PROCESSOR` 过滤后仍留在 `points` 数组中的全部点，可以使用：

```bash
--point_scope processed
```

OpenPCDet 的预处理点数组和最终硬体素中的点集合不一定相同，因此制定模型攻击预算时应使用默认的 `voxelized` 统计。

默认输出到：

```text
output/radar_attack/pointpillar_radar/feature_statistics.json
```

工具为每个特征计算：

- 有限值和非有限值数量；
- min、max、mean 和总体标准差；
- 0.1%、1%、5%、25%、50%、75%、95%、99% 和 99.9% 分位数；
- IQR（Q75-Q25）；
- robust range（Q99-Q01）。
- 不超过 32 个取值时的完整离散值集合。

均值、标准差、min 和 max 使用全部点流式精确计算。分位数默认最多对全局均匀抽取的 1,000,000 个点计算；数据点不超过该数量时是精确分位数。可用下面的参数控制：

```bash
--max_quantile_points 2000000
--point_scope voxelized
--quantiles 0.001 0.01 0.05 0.5 0.95 0.99 0.999
--num_samples 100
--output output/radar_attack/custom_statistics.json
```

特征统计提供数值尺度依据，但不能单独证明扰动物理可实现。正式 epsilon 还应结合雷达测量精度、特征单位和攻击威胁模型；尤其不能因为 `xyz`、RCS、Doppler 和时间的数值范围不同，就直接对它们使用同一个预算。

当前 VoD 5 帧数据中的 `time` 是离散 sweep 编号 `-4, -3, -2, -1, 0`，不是连续时间值。对它直接增加连续 FGSM/PGD 扰动只能作为数字特征空间基线，不能直接解释为物理可实现的时间攻击。物理约束实验应暂时排除 time，或另行设计离散的帧删除、帧替换和顺序扰动。

## 快速验证

先使用少量样本运行点云 FGSM，并跳过耗时的官方 AP：

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type fgsm \
    --attack_feature xyz \
    --epsilon_xyz 0.01 \
    --voxel_mode fixed \
    --num_samples 10 \
    --no_vod_eval \
    --extra_tag point_fgsm_smoke
```

## 点云 PGD 示例

下面的命令同时攻击空间坐标、RCS、多普勒速度和时间特征：

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type pgd \
    --attack_feature all \
    --epsilon_xyz 0.01 \
    --epsilon_rcs 0.05 \
    --epsilon_doppler 0.05 \
    --epsilon_time 0.01 \
    --pgd_steps 10 \
    --random_start \
    --voxel_mode revoxelize \
    --save_adv \
    --adv_format npy \
    --no_vod_eval \
    --extra_tag point_pgd_all
```

如果需要固定的统一 PGD 步长，可以增加 `--step_size`。未指定时，每个特征组的默认步长是：

```text
2 × 该特征组的 epsilon / pgd_steps
```

## 体素基线示例

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain voxel \
    --attack_type pgd \
    --attack_feature doppler \
    --epsilon 0.05 \
    --pgd_steps 10 \
    --no_vod_eval \
    --extra_tag voxel_pgd_doppler
```

体素攻击仅使用统一的 `--epsilon`；`epsilon_xyz` 等分组预算只对点云攻击生效。

## 扰动特征与 epsilon

`--attack_feature` 决定实际被修改的特征：

| 参数值 | 对应特征 |
| --- | --- |
| `xyz` | `x, y, z` |
| `rcs` 或 `intensity` | `rcs`、`intensity` 或 `power` |
| `doppler` | `v_r, v_r_comp` 等速度特征 |
| `time` | `time` 或 `timestamp` |
| `all` | 配置中所有能够识别的上述特征 |

点云攻击采用逐特征的 \(L_\infty\) 约束：

```text
|adversarial_feature - clean_feature| <= epsilon_for_that_group
```

各预算的含义如下：

| 参数 | 作用 |
| --- | --- |
| `--epsilon` | 未提供分组预算时使用的默认值 |
| `--epsilon_xyz` | `x, y, z` 的最大绝对扰动 |
| `--epsilon_rcs` | RCS/强度特征的最大绝对扰动 |
| `--epsilon_doppler` | `v_r, v_r_comp` 的最大绝对扰动 |
| `--epsilon_time` | 时间特征的最大绝对扰动 |

这些值作用在数据加载和特征编码后的数值上，单位跟随数据集中的对应特征。对当前 VoD 雷达配置，`xyz` 通常以米表示，速度和时间预算应根据实际数据定义与统计范围选择，不能直接把 `xyz` 的 epsilon 照搬给所有特征。

例如：

```bash
--attack_feature all \
--epsilon 0.02 \
--epsilon_xyz 0.01 \
--epsilon_doppler 0.05
```

表示 `xyz` 使用 `0.01`，Doppler 使用 `0.05`，其余被选中的特征回退到统一的 `0.02`。

## 保存对抗点云

只有点云攻击可以使用：

```bash
--save_adv --adv_format npy
```

默认保存到当前实验目录的：

```text
adversarial_points/
├── <frame_id>.npy
└── manifest.jsonl
```

也可以用 `--adv_dir` 指定输出目录，或通过 `--adv_format bin` 保存连续的 `float32` 二进制数据。

每个 `.npy` 文件的形状是：

```text
[number_of_points, number_of_point_features]
```

文件不包含 OpenPCDet 添加的 batch 索引。`manifest.jsonl` 记录帧号、形状、特征顺序、攻击参数以及最大和平均扰动，读取 `.bin` 时应以 manifest 中的 `shape` 恢复数组。

保存的对抗点云可以作为跨模型迁移攻击的输入，但目标模型必须使用兼容的坐标系、特征顺序、量纲和预处理。当前 runner 只直接加载 OpenPCDet 模型，跨架构迁移效果需要在目标模型上另行评估。

## VoD 官方评估

官方 VoD 评估默认启用。默认 devkit 路径是：

```text
~/VoD-evaluation
```

完整验证集评估示例：

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type fgsm \
    --attack_feature xyz \
    --epsilon_xyz 0.01 \
    --voxel_mode revoxelize \
    --vod_devkit ~/VoD-evaluation \
    --extra_tag point_fgsm_vod
```

如果标签不位于数据集的 `training/label_2`，可显式指定：

```bash
--vod_label_dir /absolute/path/to/training/label_2
```

官方评估会分别导出 clean 和 adversarial 的 KITTI 格式预测，并计算：

- entire area 与 ROI；
- 3D AP/mAP；
- BEV AP/mAP；
- AOS/mAOS；
- clean 到 adversarial 的绝对下降和相对下降。

使用 `--num_samples` 得到的子集 AP 只适合流程验证，不能和完整验证集 AP 直接比较。配置中的 `MODEL.POST_PROCESSING.SCORE_THRESH` 会在官方评估前删除低分预测；进行正式实验时应固定并记录该值。

当前官方 VoD 评估仅支持：

```text
--launcher none
```

不需要官方 AP 时使用：

```bash
--no_vod_eval
```

## 输出

实验结果默认位于：

```text
output/<config_group>/<config_name>/<extra_tag>/
```

主要文件包括：

```text
attack_results.json              # 完整参数和结构化结果
attack_results.txt               # 简要结果
log_attack_<type>_<time>.txt     # 运行日志
vod_predictions/
├── clean/                       # clean KITTI 格式预测
└── adversarial/                 # adversarial KITTI 格式预测
adversarial_points/              # 使用 --save_adv 时生成
```

内部指标包括 clean/attacked Recall、样本级攻击成功率、目标级攻击成功率、Recall drop，以及点云攻击的最大和平均绝对扰动。论文实验应优先报告完整验证集上的 VoD 官方 AP，而不是只使用内部攻击成功率。

## 批量基线实验

`run_experiments.py` 可以从 YAML 顺序运行一组攻击，自动分配互不冲突的输出目录，并汇总结果。仓库提供了 100 帧 epsilon 筛选配置：

```text
tools/radar_attack/configs/vod_point_baseline_screen.yaml
```

先检查将要执行的命令，不启动模型：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_point_baseline_screen.yaml \
    --dry-run
```

只运行一个实验：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_point_baseline_screen.yaml \
    --only fgsm_xyz_e001
```

确认后运行整个筛选 campaign：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_point_baseline_screen.yaml \
    --keep-going
```

每个实验输出到：

```text
output/kitti_models/pointpillar_radar/
└── vod_point_baseline_screen/
    ├── <experiment_name>/
    │   └── attack_results.json
    ├── experiment_state.json
    ├── summary.csv
    └── summary.md
```

运行器以有效且参数匹配的 `attack_results.json` 作为完成标志。命令中断后重新执行会跳过已有匹配结果；YAML 参数变化时旧结果会标记为 `stale` 并自动重跑，`--force` 会无条件重新运行选中的实验。只重新生成汇总表可使用：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_point_baseline_screen.yaml \
    --summary-only
```

汇总表中的 `epsilon_default` 是命令行 `--epsilon` 的回退值；实际提供了 `epsilon_xyz`、`epsilon_rcs`、`epsilon_doppler` 或 `epsilon_time` 时，应以相应的分组预算列为准。

筛选配置使用 `num_samples: 100`、`sample_strategy: uniform` 和 `no_vod_eval: true`，在整个验证集上均匀选取固定的 100 帧，用于快速比较 Recall/ASR。也可以使用 `sample_strategy: random` 配合 `seed` 得到可复现的随机子集；`first` 则保留旧的前 N 帧行为。筛选出代表性 epsilon 后，必须去掉样本限制，在完整 1296 帧验证集上启用 VoD 官方 AP，才能形成论文表格。

## 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--attack_domain` | `voxel` | `point` 为原始点云攻击，`voxel` 为体素基线 |
| `--attack_type` | `fgsm` | `fgsm` 或 `pgd` |
| `--attack_feature` | `all` | 选择被攻击的语义特征组 |
| `--epsilon` | `0.05` | 统一扰动预算或分组预算的回退值 |
| `--pgd_steps` | `5` | PGD 迭代次数 |
| `--step_size` | 自动 | 点云 PGD 的统一步长 |
| `--random_start` | 关闭 | 点云 PGD 随机初始化 |
| `--voxel_mode` | `fixed` | 点云攻击的 pillar 拓扑策略 |
| `--save_adv` | 关闭 | 保存原始对抗点云，仅支持 point |
| `--num_samples` | 全部 | 限制攻击样本数，用于调试 |
| `--seed` | `1024` | NumPy、PyTorch 和 CUDA 随机种子 |
| `--vod_eval` | 开启 | 运行 VoD 官方 clean/adv AP |

查看全部参数：

```bash
python tools/radar_attack/run_attack.py --help
```

## 测试

在仓库根目录运行：

```bash
python -m unittest discover -s tools/radar_attack/tests
```

测试覆盖点云输出、扰动投影、特征选择、体素 Doppler 通道、点云保存、VoD 结果汇总、预测目录清理，以及批量命令和结果汇总。正式修改攻击或体素化逻辑后，还应使用真实 checkpoint 做小样本 GPU 验证。

## 当前限制

- 当前实现是依赖模型训练损失的 untargeted white-box attack。
- `revoxelize` 对离散 pillar 分配使用 BPDA 近似，不代表真实体素化是连续可微的。
- 当前只修改已有点的属性，不包含点添加、点删除、目标级区域约束或物理可实现性约束。
- 当前 runner 面向 OpenPCDet；跨模型攻击需要单独构建目标模型适配和统一评估协议。
- FGSM/PGD 是基线，后续研究应进一步加入 4D 雷达物理约束、稀疏性约束、目标区域攻击和跨模型迁移实验。
