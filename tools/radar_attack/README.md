# 4D Radar Adversarial Attack

`radar_attack` 是面向 OpenPCDet 4D 雷达检测模型的白盒对抗攻击实验模块。目前支持在原始雷达点云上运行 FGSM、PGD 和 I-ADV-RCS，在体素特征上运行 FGSM、PGD，并比较攻击前后的检测指标和 View-of-Delft（VoD）官方 AP。

本目录不依赖旧的 `tools/attacks` 实现，但仍依赖 OpenPCDet 的数据集、模型、损失函数和配置系统。

截至 2026-08-06 的方法、实验结果、问题和下一阶段路线汇总见
[`docs/research_status_report_2026-08-06.md`](docs/research_status_report_2026-08-06.md)。

## 目录结构

```text
tools/radar_attack/
├── run_attack.py                 # 推荐的命令行入口
├── runner.py                     # 加载数据和模型，执行攻击、推理与评估
├── analyze_features.py           # 模型输入特征分布与 IQR
├── diagnose_object_evidence.py   # 对象 loss 与逐目标梯度敏感度诊断
├── analysis/
│   ├── feature_stats.py          # 流式特征统计
│   └── object_evidence.py        # 候选证据、敏感度和诊断报告
├── attacks/
│   ├── base.py                   # 统一的 AttackOutput
│   ├── gradient.py               # 原始点云 FGSM/PGD
│   ├── iadv.py                    # LiDAR I-ADV 到 Radar RCS 的透明复现
│   └── voxel.py                  # 体素 FGSM/PGD 对照基线
├── adapters/
│   └── openpcdet.py              # 可微分点云到 PointPillars 体素适配
├── evaluation/
│   ├── metrics.py                # 扰动、Recall 和攻击成功率
│   ├── storage.py                # 保存对抗点云与 manifest
│   └── vod.py                    # VoD 官方 AP 评估适配
└── tests/
    ├── test_components.py        # 攻击组件测试
    └── test_object_diagnostics.py # 对象诊断单元测试
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

### Radar Measurement-Consistent Geometry Attack

`--attack_space radar_measurement` 启用 Radar 测量参数化的对象级几何攻击。当前
VoD Radar 坐标约定为 `x` 向前、`y` 向左、`z` 向上，内部使用：

```text
range     = sqrt(x^2 + y^2 + z^2)
azimuth   = atan2(y, x)
elevation = atan2(z, sqrt(x^2 + y^2))
```

PGD/FGSM 的主变量是 `range/azimuth/elevation`，而不是三个相互独立的
Cartesian 坐标。每次迭代都会由更新后的测量值确定性重建 XYZ，再进行真实硬
体素化并送入 PointPillars；离散 pillar 分配的反向传播仍使用现有 BPDA/
straight-through 近似。

第一版威胁模型为：

```text
Digital-domain, white-box, object-level attack.
Only existing target points from the current Radar sweep are perturbed.
Historical sweeps remain part of the five-frame detector input but are unchanged.
```

实际攻击 mask 是“目标框内点、`time == 0` 当前 sweep、clean 硬体素化实际保留
的点”三者的交集。点数、点顺序、batch index、RCS、`v_r`、`v_r_comp` 和
`time` 均保持不变。当前数据加载路径没有提供逐帧 ego-velocity 向量和可复现
的官方 `v_r_comp` 计算，因此 angle 改变后暂时保留 clean `v_r_comp`，不自行
猜测补偿公式。这意味着当前只保证几何测量参数化的一致性。

第一版只支持：

```text
--attack_domain point
--attack_feature xyz
--point_scope gt_boxes
--voxel_mode revoxelize
```

它不会独立裁剪重建后的 XYZ。若一个 measurement 更新使点越出 detector 的
`POINT_CLOUD_RANGE`，程序会在 measurement space 中逐次减半该更新；仍无法
得到合法点时回退到 clean measurement，从而避免破坏
`xyz = T(range, azimuth, elevation)`。

小规模工程 smoke test 示例：

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_space radar_measurement \
    --attack_type pgd \
    --attack_feature xyz \
    --point_scope gt_boxes \
    --target_classes Car \
    --point_target_selection all_gt \
    --attack_loss training \
    --epsilon_range 0.05 \
    --epsilon_azimuth_deg 0.10 \
    --epsilon_elevation_deg 0.10 \
    --pgd_steps 2 \
    --voxel_mode revoxelize \
    --num_samples 2 \
    --no_vod_eval \
    --extra_tag measurement_geometry_smoke
```

角度配置和日志使用 degrees，内部优化统一转换为 radians。分别使用
`--step_size_range`、`--step_size_azimuth_deg` 和
`--step_size_elevation_deg` 可覆盖默认的 `2 * epsilon / pgd_steps`。上述预算
只是用于代码验证的保守工程值，不是经过验证的传感器误差或物理攻击上限。

结果会记录三个 measurement 最大扰动、Cartesian 最大/平均 L2 位移、距离分桶
位移、当前 sweep 有效攻击点数、越界回退数，以及历史点、非目标点和非几何
特征的修改数；后三类修改数必须为 0。

这是 measurement-consistent digital geometry attack，不是 waveform/IQ-level
攻击，也不能据此声称已经具有完整物理可实现性。

#### Stage 2 Q95 归一化几何异常度

所有 point-domain 攻击还会对攻击前后同索引点计算：

```text
x_r     = |delta range|
x_alpha = mean_range * cos(mean_elevation) * |wrapped delta azimuth|
x_theta = mean_range * |delta elevation|
```

三项单位均为米。程序只使用 Stage 2 clean matched-pair statistics 的
`d_xyz <= 1 m` gate，并按 clean 点自身 range 查询 `0-10/10-20/20-30/30-50 m`
桶中的 `abs_delta_r/delta_s_az/delta_s_el` P95：

```text
z_r     = x_r / Q95_r(range_bin)
z_alpha = x_alpha / Q95_alpha(range_bin)
z_theta = x_theta / Q95_theta(range_bin)
A       = max(z_r, z_alpha, z_theta)
```

默认参考被固化在
`tools/radar_attack/references/vod_stage2_gate1_q95.json`，包含源
`point_quantiles.csv` 的 SHA256。加载时强制检查 `gate_m == 1.0` 和
`quantile == 0.95`，防止误用其他 gate。可用
`--measurement_q95_reference` 指定同格式参考文件，但仍必须是 1 m gate。

逐点结果保留完整 `(z_r,z_alpha,z_theta)`，输出为：

```text
measurement_naturalness_points.csv.gz
```

汇总结果同时报告三个 z 的 P95、A 的 P50/P95/P99/max、`A>1/2/5` 比例、
最异常维度占比，并分别统计所有参考覆盖点与实际发生几何变化的覆盖点。
clean range 超出 `0-50 m` 的点标为未覆盖，不使用全局 Q95 偷偷回填。

注意：Stage 2 原统计按相邻 Car 目标帧对的平均 GT 中心 range 分桶；这里按
用户定义使用攻击点自身 clean range 查桶。对象框内点二者接近，但对背景点及
其他类别，这是一个需要在论文中明确声明的参考尺度迁移，而不是传感器概率模型。

#### 多 sweep 时序预处理

`prepare_temporal.py` 为多 sweep measurement attack 恢复数据层元信息；三个
非 `none` 的 `--temporal_mode` 都会在 FGSM/PGD 中使用这些信息：

1. 将五帧文件中 `time=-k` 的点与 `frame_id-k` 的单帧 Radar 点按原始顺序
   对齐；
2. 由对应 XYZ 拟合 source Radar 到 reference Radar 的刚体变换，并缓存为
   `.npz`；
3. 在 source sweep 坐标系读取该帧 tracking GT 框，只保留 reference frame
   中同一 tracking ID 的目标；
4. 按归一化目标局部中心距离分配重叠框中的点；
5. 验证 source XYZ → range/azimuth/elevation → source XYZ → reference XYZ
   的零扰动往返误差，同时要求 RCS、Doppler、time 不变。

运行完整验证集预处理：

```bash
/home/car/anaconda3/envs/openpcdet/bin/python \
    tools/radar_attack/prepare_temporal.py \
    --dataset_root data/view_of_delft \
    --split val
```

快速检查可以增加 `--max_frames 10`。默认缓存和报告分别写入：

```text
output/radar_attack/temporal_sweep_cache/<frame_id>.npz
output/radar_attack/temporal_sweep_cache/preparation_report.json
```

缓存加载时仍会重新核对 source/reference 点数、RCS/Doppler 顺序和刚体重建
误差，因此数据或点顺序变化不会被静默接受。场景起始位置直接服从五帧文件中
实际存在的 sweep id，不会越过 clip 边界补齐历史帧。

当前阶段使用 tracking GT，符合现有白盒、GT-object 数字域威胁模型，但属于
oracle object association。后续若研究更现实的攻击知识约束，需要将其替换为
检测/跟踪关联。由于尚未耦合 Doppler 和几何，这一阶段应称为
track-consistent multi-sweep measurement-space geometry attack，不能称为
waveform 或完整物理 Radar 攻击。

三种多 sweep 参数化使用完全相同的跟踪目标点、measurement 预算、loss、真实
hard revoxelization 和坐标变换，只改变参数共享粒度：

- `point_independent`：每个目标点独立使用一组
  `[delta_range, delta_azimuth, delta_elevation]`；
- `object_per_sweep`：同一 tracking ID 在同一个 sweep 内共享一组参数，不同
  sweep 可以不同；
- `track_shared`：同一 tracking ID 的所有可用 sweep 共享一组参数。

measurement 更新在各点自己的 source Radar 坐标系完成，再通过缓存刚体变换
返回 reference frame。若组内任一点越过 detector 范围，整个参数组共同回退：
独立点模式只回退该点，object-per-sweep 回退该对象在该 sweep 的点，
track-shared 回退整条轨迹。输出中的 `Temporal parameter groups` 可用于核对三种
模式的实际自由度。

`point_independent` 还支持 `--point_scope scene`，用于全场景 Cartesian
压力测试的测量空间对照。该组合为每个 clean hard-voxel active 场景点分配独立
measurement 参数；背景点没有对象或轨迹身份，因此 scene scope 不支持
`object_per_sweep` 或 `track_shared`。对象级主实验仍应使用 `gt_boxes`。

一帧、10 步 PGD 示例：

```bash
cd /home/car/OpenPCDet/tools
/home/car/anaconda3/envs/openpcdet/bin/python \
    radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt ../checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_space radar_measurement \
    --temporal_mode track_shared \
    --attack_type pgd \
    --attack_feature xyz \
    --point_scope gt_boxes \
    --target_classes Car \
    --point_target_selection all_gt \
    --attack_loss training \
    --epsilon_range 0.05 \
    --epsilon_azimuth_deg 0.10 \
    --epsilon_elevation_deg 0.10 \
    --pgd_steps 10 \
    --voxel_mode revoxelize \
    --num_samples 1 \
    --no_vod_eval \
    --extra_tag temporal_track_shared_smoke
```

把示例中的 `track_shared` 分别替换为 `point_independent` 或
`object_per_sweep` 即可运行另外两种模式。不提供非 `none` 的 temporal mode 时
仍执行原来的 current-sweep measurement attack。三种多 sweep 模式都支持
`training`、`object_evidence`、`object_hybrid` 和 `object_iou_s` loss；后三者仍要求
`--point_target_selection clean_detected`。

#### IoU-S 原论文实验线 A：全场景 point perturbation

`--attack_type iou_s_original` 独立复现作者公开代码中的 `iou_per` 分支，不能与
下面的 Radar 对象级适配 loss 混称。默认行为与公开实现对齐：

- 整帧所有点均可修改，但严格保持点数、顺序和非 XYZ 特征不变；
- 每一步真实重体素化，并动态使用当前 post-NMS 预测框和分数；
- 每个有效 GT 与每个当前预测组成一对，最小化
  `-log(1-score)-log(1-IoU3D)` 的总和，不按类别筛选或均衡；
- 使用 Adam，默认 500 步、学习率 `0.01`、正向均匀 XYZ 初始噪声 `0.01`；
- 距离项是双向 squared Chamfer 与全局 XYZ L2 之和，默认权重为 1；
- 不使用 GT 框点掩码，也不执行逐点 epsilon 硬投影。

OpenPCDet 的 hard voxel assignment 仍是离散操作；每一步按当前 XYZ 重建
topology，梯度通过实际进入 voxel 的 point gather 回传。旋转 3D IoU 使用本目录
已有的分段可微 PyTorch 实现。`--target_classes` 只决定最终 Object Failure ASR
评估哪些类别，不会限制原始 IoU-S loss 中参与优化的 GT 类别。

100 帧论文参数筛选：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iou_s_original_screen.yaml
```

为诊断原论文的联合最优样本保存规则，可在同一批 100 帧上返回第 500
步 Adam 结果。该配置不改变优化轨迹或损失，只把返回策略从默认的
`joint_best` 改为 `last`：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iou_s_original_last_screen.yaml
```

诊断输出同时报告平均最佳步骤、前 10/100 步内选中比例、联合最优与最后
一步的每对预测损失，以及相邻迭代后处理框的保持率、切换率和数量变化率。

筛选稳定后运行完整 1296 帧和 VoD 官方 AP：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iou_s_original_full.yaml
```

直接命令形式：

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type iou_s_original \
    --attack_feature xyz \
    --attack_space feature \
    --voxel_mode revoxelize \
    --point_scope scene \
    --point_target_selection all_gt \
    --target_classes Car Pedestrian Cyclist \
    --iou_s_original_steps 500 \
    --iou_s_original_lr 0.01 \
    --iou_s_original_init_noise 0.01 \
    --iou_s_original_distance_weight 1.0 \
    --iou_s_original_log_epsilon 1e-8 \
    --iou_s_original_chamfer_chunk_size 1024 \
    --num_samples 100 \
    --sample_strategy uniform \
    --no_vod_eval \
    --extra_tag iou_s_original_screen
```

这里的通用 `--epsilon` 和 `--pgd_steps` 不参与攻击。实际扰动代价由输出中的
最大/平均 XYZ 位移、Chamfer/L2 距离和修改点数审计。参考实现：
[haichen-ber/IoU-S-Attack](https://github.com/haichen-ber/IoU-S-Attack)。

##### Car/current-scan Q95 几何约束

`--iou_s_geo_weight` 在实验线 A 的完整全场景攻击变量上增加 Radar 几何软约束，
不会缩小可攻击点范围。参考掩码在 clean 点云上冻结，只包含 Car GT 框关联、
`time=0` 且 clean range 位于 `0--50 m` 的点。框外点、其他类别点、历史 sweep
和参考距离外的点仍由原始 IoU-S 优化，但不使用 Car/current-scan 统计约束。

对参考点计算三个 measurement displacement 的 Stage 2 Q95 归一化值，并使用：

```text
L_geo = mean_over_reference_points(
    relu(z_range - 1)^2
  + relu(z_azimuth - 1)^2
  + relu(z_elevation - 1)^2
)
L_total = L_IoU-S-detection + iou_s_geo_weight * L_geo
```

Q95 sweep 配置明确设置 `iou_s_original_distance_weight: 0`，因此不会计算或使用
双向 Chamfer 和全局 XYZ L2；此时 `lambda_g=0` 是无距离正则的 IoU-S 检测
loss 基线，不再是原论文完整目标。`L_geo` 的分母严格是参考点数，不是全场景
点数；没有合法参考点的场景令该项为零。默认参考
`vod_stage2_gate1_point_range_q95.json` 来自 1 m MNN clean pairs，并按 clean
点自身 range 重新分桶。

每次运行会输出 `iou_s_original_loss_trace.csv`，逐帧、逐步保存检测 loss、
每对预测 loss、距离项、原始/加权几何项和总 loss。先运行 10 帧 lambda sweep：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iou_s_geo_q95_screen.yml
```

其中比较 `iou_s_geo_weight = 0, 0.01, 0.1, 1`。应以轨迹中的
`weighted_geometry_loss` 相对 `iou_s_detection_loss` 的实际比例判断权重，而
不能只比较名义 lambda。

#### Radar Object IoU-S loss（适配实验线，非原论文复现）

`--attack_loss object_iou_s` 是针对当前 Radar 对象级漏检任务接入的 IoU-S
优化目标。它不会改变 measurement attack 的扰动变量、点掩码或重体素化流程，
只替换用于求梯度的目标函数。对干净输入中检测成功的目标，先固定 GT 附近的
pre-NMS anchors，并按干净 3D IoU（score 用于打破并列）保留前
`--iou_s_candidate_topk` 个候选。每个候选的梯度上升目标为：

```text
L = w_score * log(1 - sigmoid(class_logit) + eps)
  + w_iou   * log(1 - IoU3D(decoded_box, GT) + eps)
```

因此对 `L` 做梯度上升会同时降低目标类别置信度和预测框与 GT 的 3D IoU。
默认 `w_score=w_iou=1`。候选先在每个目标内取平均；选择多类目标时，再按
“同类目标平均、所选类别平均”聚合，Car 不会因为数量更多而支配 loss。

OpenPCDet 自带 rotated-IoU CUDA overlap 只提供前向 BEV overlap，不能为 x/y/yaw
提供完整梯度。本实现使用纯 PyTorch 的 yaw-aware 旋转矩形求交和高度交集；交集
拓扑与顶点排序是离散的，但固定拓扑内对中心、尺寸和 yaw 分段可微。候选身份在
clean forward 后保持不变，避免 PGD 过程中因 NMS 或候选切换导致优化目标漂移。

这与论文官方代码动态使用 post-NMS proposals 的写法并不逐行相同，而是适配
OpenPCDet PointPillars 和 Radar 测量空间的稳定实现。它仍是白盒、数字域的代理
loss，不能单独证明物理可实现性。

10 帧工程验证：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_radar_object_iou_s_smoke.yaml
```

该配置攻击 Car、Pedestrian、Cyclist 三类 clean-detected 对象，使用当前 sweep
对象点、10 步 PGD、`range=0.5 m`、方位角/俯仰角各 `1 degree`，并执行真实
revoxelization。预算沿用目前的研究性数字域实验设置，不代表已经验证的 Radar
测量误差上限。核心参数也可直接在 `run_attack.py` 中设置：

```text
--attack_loss object_iou_s
--iou_s_candidate_topk 32
--iou_s_score_weight 1.0
--iou_s_iou_weight 1.0
--iou_s_log_epsilon 1e-6
```

#### Current-sweep 公平对照与 100 帧机制筛选

`--current_sweep_only` 让 feature-space XYZ 攻击使用与 measurement
attack 完全相同的点集合：clean-detected GT 目标点、`time == 0`、且被
clean hard voxelization 保留。旧 five-sweep XYZ 行为仍然保留；只有显式
加入该参数时才启用公平 current-sweep baseline。

固定五组主实验加一个 displacement-matched XYZ 辅助点的 100 帧配置为：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_measurement_geometry_screen.yaml
```

它比较 `XYZ-current`、range-only、azimuth-only、elevation-only 和三个
measurement 变量联合攻击。五组使用相同 uniform 子集、seed、training loss、
10 步 PGD、确定性初始点和真实 hard revoxelization。当前 measurement 预算是
`research/digital budget`，不是经过验证的物理传感器不确定性。

每个实验另外输出：

```text
current_sweep_targets.csv
current_sweep_summary.yaml
current_sweep_summary.md
object_endpoint_metrics.csv
target_correlations.json
```

逐目标 CSV 包含总点数/current 点数/active current 点数、IoU/score/evidence
变化、Cartesian 位移和真实 hard-pillar reassignment。campaign 根目录中的
`comparison_summary.csv` 和 `comparison_summary.md` 是五种主攻击及一个辅助
XYZ 预算的紧凑表格。

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

## 对象级梯度诊断

`diagnose_object_evidence.py` 用于回答以下问题，而不是生成正式对抗样本：

- 对象证据 loss 是否比检测器训练 loss 更能降低目标置信度；
- geometry、Doppler 和 RCS 的 IQR 归一化与实际 probe-budget
  归一化梯度敏感度；
- 不同目标的敏感度比例是否不同；
- 敏感度与距离、模型有效点数、速度和类别的关系。

第一版仅支持当前 `PointPillars + AnchorHeadSingle`。它先用正常后处理筛选干净检测成功的 GT，然后在干净输入上固定该目标附近的 pre-NMS anchor 候选。训练 loss 沿梯度上升，对象 loss 定义为候选目标类别 `LogSumExp` 证据的相反数，因此沿梯度上升会压低整组候选证据。

先运行 1 帧 GPU 冒烟：

```bash
/home/car/anaconda3/envs/openpcdet/bin/python \
    tools/radar_attack/diagnose_object_evidence.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /home/car/OpenPCDet/output/kitti_models/pointpillar_radar/vod5_retrain_seed0/ckpt/checkpoint_epoch_80.pth \
    --feature_stats /home/car/OpenPCDet/output/radar_attack/pointpillar_radar/feature_statistics.json \
    --target_classes Car Pedestrian Cyclist \
    --num_samples 1 \
    --sample_strategy uniform \
    --workers 0 \
    --output_dir /home/car/OpenPCDet/output/radar_attack/pointpillar_radar/object_evidence_dual_scale_smoke
```

通过后运行均匀抽样的 100 帧诊断：

```bash
/home/car/anaconda3/envs/openpcdet/bin/python \
    tools/radar_attack/diagnose_object_evidence.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /home/car/OpenPCDet/output/kitti_models/pointpillar_radar/vod5_retrain_seed0/ckpt/checkpoint_epoch_80.pth \
    --feature_stats /home/car/OpenPCDet/output/radar_attack/pointpillar_radar/feature_statistics.json \
    --target_classes Car Pedestrian Cyclist \
    --num_samples 100 \
    --sample_strategy uniform \
    --workers 4 \
    --output_dir /home/car/OpenPCDet/output/radar_attack/pointpillar_radar/object_evidence_dual_scale_100
```

默认的类别匹配阈值为 `Car=0.5`、`Pedestrian=0.25`、`Cyclist=0.25`，可使用下面的参数覆盖：

```bash
--iou_thresholds Car=0.5 Pedestrian=0.25 Cyclist=0.25
```

一步置信度探测默认使用每列 `1% IQR`，同时限制 XYZ 每列最多 `0.02 m`、Doppler 每列最多 `0.1 m/s`、RCS 最多 `0.2`。这些值仅用于比较两个 loss 的局部方向，不是正式攻击预算。默认 `--probe_voxel_mode fixed` 与求梯度时的柱拓扑一致；`revoxelize` 会额外混入不可微的换柱效应，应作为单独诊断而不是替代默认结果。

输出目录包含：

```text
per_object.csv                 # 每个干净检测且存在模型有效点的目标
summary.json                   # 汇总、类别统计和 Spearman 相关性
run_config.json                # 完整命令参数
diagnose_object_evidence.log   # 运行日志
```

`iqr_*_sensitivity_sum` 表示“若每个特征都允许移动一个 IQR，哪个域更敏感”，适合比较数据尺度；`budget_*_sensitivity_sum` 使用本次经过 cap 后的实际局部 probe 步长，表示“在这次真正允许的微小改变量下，哪个域的一阶影响更大”，它才是预算分配的主要依据。`budget_*_allocation` 是三个域的 budget sensitivity 除以总和。为保证旧结果可追溯，无前缀字段仍等于 `iqr_*`，不能再把它当作实际预算分配。`*_mean_per_point` 用于消除目标点数影响。

同一诊断现在还直接测量 `object_hybrid` 的分类上升梯度
`g_cls=∇(-evidence)` 与定位上升梯度 `g_loc=∇L_loc`。每个目标会分别对
geometry、Doppler 和 RCS 输出：

- `hybrid_*_classification_l1` 与 `hybrid_*_localization_l1`：目标框内实际
  进入模型的点上的梯度 L1 范数；
- `hybrid_*_balance_beta_l1=||g_cls||₁/(||g_loc||₁+δ)`：让两项达到同一
  L1 尺度所需的逐目标权重；
- `hybrid_*_cosine_similarity` 与 `hybrid_*_sign_agreement`：两项目标是
  协同还是冲突；
- `hybrid_*_beta1_sign_change_fraction`：固定 `β=1` 相对纯分类梯度真正
  改变了多少 sign-PGD 更新方向；
- `hybrid_*_balanced_sign_change_fraction`：改用上述逐目标平衡权重后的
  对应比例。

这些字段测量的是“某目标自身的 loss 对该目标框内点”的局部关系。它不把
多个目标的梯度混在一起，因此适合判断固定全局 β 是否合理；正式的多目标
攻击仍需另外处理不同目标之间的梯度冲突。

当前 geometry 仍是模型输入中的 Cartesian XYZ 分析，不应解释为物理可实现的 Radar range/angle 攻击；Doppler 也会另外报告 `v_r` 与 `v_r_comp` 的耦合敏感度。诊断脚本仍读取全部五帧输入点，`time=0` 只应在未来定义“只改当前 sweep”的攻击掩码时使用，不是本次尺度修正的一部分。

## 对象级攻击损失

点云 FGSM/PGD 现在支持 `--attack_loss object_evidence`。它先在干净输入中筛选检测成功的目标，固定目标附近的 pre-NMS anchor 候选，再通过最小化这些候选的目标类别 LogSumExp 证据来生成梯度。攻击实现采用梯度上升，因此代码中的目标函数是“负对象证据”。I-ADV 保持原论文训练 loss，不混入这个修改，以免失去迁移基线含义。

在连续终点诊断确认分类证据下降与 IoU 下降存在分工后，新增
`--attack_loss object_hybrid`。其每目标优化目标由两部分组成：

- 分类项沿用负 LogSumExp 对象证据，压低目标类别分数；
- 定位项在干净输入上，从目标区域选择 decoded-box IoU 最高的固定 top-k
  anchors，以干净类别置信度 softmax 作为固定权重，增大当前框编码与 GT
  框编码之间的 Smooth L1 误差；yaw 使用周期化正弦差。

两部分先在每个目标内部计算再跨目标平均，防止一个目标仅因候选数量更多而
支配 loss。定位候选身份和权重在迭代中保持不变，避免优化过程通过切换
anchors 逃避定位项。主要参数为：

```text
--hybrid_localization_weight 1.0
--hybrid_localization_topk 32
```

`object_evidence` 保持原实现，因而三方对照只需切换 `--attack_loss
{training,object_evidence,object_hybrid}`。当前新目标仍只支持
`AnchorHeadSingle`，它是方法探索组件，还不能单独称为最终 Radar-aware
方法。

公平比较对象 loss 与训练 loss 时，两者都必须使用：

```text
--point_scope gt_boxes
--point_target_selection clean_detected
```

这样两边使用同一组干净检测成功的 GT、同一批可修改点、相同预算和初始化，唯一变量才是 loss。对象 loss 当前只支持 `AnchorHeadSingle`；`all_gt` 仍是旧的默认行为。

可恢复的 100 帧消融配置为：

```bash
/home/car/anaconda3/envs/openpcdet/bin/python \
    tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_object_loss_ablation.yaml
```

配置中分别对 RCS 和 XYZ 比较训练 loss 与对象 loss，不使用随机起点，也不运行官方 AP。这只是验证目标级 loss 能否把局部诊断优势转化为 Object ASR，不应直接称为最终方法。

### 对象级成功率与连续终点指标

评估只把 clean 中正确检测到的所选 GT 作为分母，并对预测框做一对一匹配。
攻击后的每个目标只属于下面一个互斥结果：

- `still_correct`：仍有同类预测通过该类的严格 IoU 阈值；
- `misclassification`：没有正确检测，但有异类预测通过严格 IoU 阈值；
- `localization_failure`：没有严格匹配，但仍有预测框与 GT 正 IoU 重叠；
- `pure_hiding`：没有任何达到分数阈值且与 GT 正 IoU 重叠的预测框。

主指标 `Pure Hiding ASR` 是 `pure_hiding / clean eligible objects`；同时输出
`Object Failure ASR`（后三种失败之和）、误分类率、定位失败率和仍正确率。
四个比例之和为 1，不再输出场景级 sample ASR。JSON 在
`metrics.object_outcomes.by_class` 保存详细分类结果，日志和 campaign 汇总只保留
总体指标以及紧凑的分类组成。

默认严格阈值为 `Car=0.5`、`Pedestrian=0.25`、`Cyclist=0.25`。可用
`--target_classes Car Pedestrian Cyclist` 选择一个或多个攻击目标类，并用
`--object_iou_thresholds Car=0.5 Pedestrian=0.25 Cyclist=0.25` 覆盖阈值。
`--object_score_threshold` 默认继承模型的 `MODEL.POST_PROCESSING.SCORE_THRESH`。

每次评估还会生成 `object_endpoint_metrics.csv`，逐个记录干净检测成功目标的：

- clean/adv 最大同类 IoU、IoU drop 和距离阈值的 margin；
- 与最大 IoU 框对应的 clean/adv score 和 score drop；
- 固定 clean 候选集合上的对象证据及其下降量；
- 预测框中心位移、中心误差增量、尺寸 L1 变化和周期化 yaw 变化。

`attack_results.json` 的 `metrics.object_endpoint_metrics` 会汇总每项的均值、
中位数、10%/90% 分位数和正值比例；campaign 的 `summary.csv` 与
`summary.md` 会列出主要连续指标。没有 adv 同类预测框时，IoU 和 score
按零计，无法定义的框位移字段在逐目标 CSV 中留空。

旧的 loss 消融结果不会自动包含这些字段。使用下面的新 campaign 可在新
目录中保留旧结果并重新运行四组公平对照：

```bash
/home/car/anaconda3/envs/openpcdet/bin/python \
    tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_object_endpoint_ablation.yaml
```

混合 loss 的训练 loss、分类对象 loss、混合对象 loss 三方消融使用：

```bash
/home/car/anaconda3/envs/openpcdet/bin/python \
    tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_object_hybrid_ablation.yaml
```

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

## I-ADV-RCS 复现

I-ADV-RCS 只修改体素化之前原始 Radar 点云的 RCS，保持 batch index、点数、点顺序、xyz、Doppler 和 time 不变。执行顺序为：

1. 使用 KD-tree 邻域和 PCA 局部法向量计算角度特征，并结合距离形成反射特征；
2. 每轮计算 OpenPCDet 检测损失对 RCS 的梯度，按样本归一化并累积 MI-FGSM 动量；
3. 使用反射特征增强梯度；
4. 按边长 `0.1 m` 的立方体比较正、负梯度极值，同组 RCS 使用统一更新方向；
5. 将 RCS 投影回 clean RCS 的 `epsilon_rcs` 邻域。

在 `gt_boxes` 攻击中，每个点先分配到一个独立目标。若一个点同时位于多个框内，使用“点到框中心的局部坐标距离除以该框半尺寸”得到归一化距离，并归给距离最小的框。默认的 `--iadv_neighbor_scope object` 会将 KD-tree、PCA 和梯度融合全部限制在同一 `(batch_id, object_id)` 内，不允许不同目标共享邻居或融合组。

论文主实验是 object-level attack。下面的命令攻击所有 Car GT 3D 框内、实际进入模型体素的 Radar 点：

```bash
python tools/radar_attack/run_attack.py \
    --cfg_file cfgs/kitti_models/pointpillar_radar.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_80.pth \
    --attack_domain point \
    --attack_type iadv \
    --attack_feature rcs \
    --epsilon_rcs 0.3 \
    --iadv_steps 10 \
    --iadv_scope gt_boxes \
    --iadv_neighbor_scope object \
    --target_classes Car \
    --iadv_attack_voxel_size 0.1 \
    --iadv_mu 1.0 \
    --iadv_lambda 1000 \
    --iadv_d_max 75 \
    --iadv_k_neighbors 16 \
    --iadv_min_neighbors 3 \
    --iadv_gradient_norm l1 \
    --object_iou_thresholds Car=0.5 \
    --save_adv \
    --no_vod_eval \
    --num_samples 10 \
    --extra_tag iadv_rcs_smoke
```

使用 `--iadv_scope scene` 可运行论文单独讨论的 full-scene attack。I-ADV 默认步长是 `0.2 * epsilon_rcs`；可以用 `--step_size` 显式覆盖。`--iadv_rcs_min` 和 `--iadv_rcs_max` 可施加数据域边界，但在没有 Radar 测量依据时默认不裁剪绝对 RCS，只执行相对 clean 点云的预算投影。

邻域/融合范围有三种模式：

| 参数值 | 含义 |
| --- | --- |
| `object` | 默认；KD-tree、PCA 和立方体融合逐 Car 隔离，只能与 `--iadv_scope gt_boxes` 一起使用 |
| `attack_union` | 保留第一版实现：同一帧所有被攻击点共享邻域和融合分组，仅用于追溯旧结果 |
| `scene` | 使用同一帧实际进入模型的点作为 PCA 邻域，融合按场景进行；全场景实验应同时设置 `--iadv_scope scene --iadv_neighbor_scope scene` |

结果 JSON 的 `metrics.attack_diagnostics` 以及 campaign 汇总表会记录有效目标数、每目标平均点数、PCA 回退率、单点组比例、每目标分组数、每组点数、跨目标邻居数和非有限梯度步数。`object` 模式发现跨目标邻居时会直接报错，而不是继续产生不可解释结果。

原论文没有公开代码，也没有说明 KD-tree 邻居数、梯度范数、PCA 退化处理、目标点提取和裁剪顺序。本实现把这些选择显式化：默认 `k=16`（包含查询点）、L1 梯度归一化、少于 3 个邻居时使用 distance-only 中性回退、GT 3D 框内点，以及先预算投影再做可选 RCS 范围裁剪。这些属于透明复现假设，不是作者确认的参数。完整迁移记录见 `docs/iadv_to_4d_radar.md`。

该命令以 Car 为目标，统一报告 `Pure Hiding ASR`、`Object Failure ASR` 及失败原因组成。若要同时攻击三类，改为 `--target_classes Car Pedestrian Cyclist`；当前对象损失仍按目标实例平均，因此这一步只是目标选择和评估扩展，还没有实现“先类内平均、再类间平均”的类别均衡损失。

旧的直接迁移配置保留为 `attack_union` 结果追溯基线：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iadv_rcs_screen.yaml \
    --dry-run
```

该配置为 PGD 设置 `point_scope: gt_boxes`，因此 PGD 与 I-ADV 修改同一类 GT Car 框内点。普通 FGSM/PGD 命令仍默认 `--point_scope scene`；需要目标区域基线时显式使用：

```bash
--point_scope gt_boxes --target_classes Car
```

下一阶段使用两份互不覆盖的配置。先运行逐车隔离的一致性实验：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iadv_rcs_object_fidelity.yaml \
    --keep-going
```

确认三组旧预算均正常后，再运行 Radar RCS 预算与分组尺度实验：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iadv_rcs_radar_screen.yaml \
    --keep-going
```

第二份配置包含 `epsilon_rcs={0.8,1.6,3.1}` 的 object-level PGD/I-ADV 对照，以及固定 `epsilon_rcs=1.6`、攻击分组边长 `{0.1,0.3,0.5,1.0} m` 的 I-ADV 尺度实验。这里改变的是攻击算法自己的立方体分组，不是 PointPillars 的模型 pillar 尺寸。两份配置都只跑均匀抽样的 100 帧并关闭官方 AP；只有 Object ASR 非零且预算不超过约 10% IQR 后，才进入完整 1296 帧官方 VoD AP。

当上述筛选门槛满足后，使用正式配置在完整验证集上比较同预算 PGD 和 I-ADV：

```bash
python tools/radar_attack/run_experiments.py \
    tools/radar_attack/configs/vod_iadv_rcs_full_ap.yaml \
    --keep-going
```

该配置固定 `epsilon_rcs=1.6` 和 I-ADV 原始 `0.1 m` 分组尺度，不设置 `num_samples`，并显式启用本地 VoD devkit 的官方 AP。它只有在 100 帧筛选完成后才创建，不能用来反向挑选预算。

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
| `--epsilon_range` | measurement attack 的距离预算，单位 m |
| `--epsilon_azimuth_deg` | measurement attack 的方位角预算，单位 degree |
| `--epsilon_elevation_deg` | measurement attack 的俯仰角预算，单位 degree |
| `--measurement_q95_reference` | 固定为 Stage 2 `1 m gate` 的 Q95 参考 JSON |

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
| `--attack_type` | `fgsm` | `fgsm`、`pgd`、`iadv` 或全场景原版 `iou_s_original` |
| `--attack_feature` | `all` | 选择被攻击的语义特征组 |
| `--epsilon` | `0.05` | 统一扰动预算或分组预算的回退值 |
| `--pgd_steps` | `5` | PGD 迭代次数 |
| `--step_size` | 自动 | 点云 PGD 的统一步长 |
| `--random_start` | 关闭 | 点云 PGD 随机初始化 |
| `--voxel_mode` | `fixed` | 点云攻击的 pillar 拓扑策略 |
| `--target_classes` | `Car` | 可同时选择 Car、Pedestrian、Cyclist |
| `--object_iou_thresholds` | `0.5/0.25/0.25` | 按类别设置 clean eligibility 与严格匹配阈值 |
| `--object_score_threshold` | 模型后处理阈值 | 对象结果评估使用的预测分数下限 |
| `--attack_loss` | `training` | 可选训练损失、对象证据、混合定位或 Radar Object IoU-S |
| `--iou_s_candidate_topk` | `32` | 每个 IoU-S 目标固定保留的 clean 高 IoU 候选数 |
| `--iou_s_score_weight` | `1.0` | IoU-S 置信度抑制项权重 |
| `--iou_s_iou_weight` | `1.0` | IoU-S yaw-aware 3D IoU 抑制项权重 |
| `--iou_s_original_steps` | `500` | 原论文 point perturbation 的 Adam 步数 |
| `--iou_s_original_lr` | `0.01` | 原论文 point perturbation 的 Adam 学习率 |
| `--iou_s_original_init_noise` | `0.01` | 原论文正向均匀 XYZ 初始噪声幅度 |
| `--iou_s_original_distance_weight` | `1.0` | 原论文 Chamfer 与全局 XYZ L2 距离项权重 |
| `--iou_s_geo_weight` | `0.0` | Car/current-scan Stage 2 Q95 几何平方 hinge 权重 |
| `--iou_s_geo_reference` | point-range 1 m MNN JSON | IoU-S 几何约束使用的 clean Q95 参考 |
| `--iadv_neighbor_scope` | `object` | I-ADV 的逐车、旧攻击点并集或全场景邻域/融合范围 |
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
- 当前只修改已有点的属性，不包含点添加或点删除；I-ADV-RCS 支持 GT 框目标区域，但仍不代表物理可实现攻击。
- 当前 runner 面向 OpenPCDet；跨模型攻击需要单独构建目标模型适配和统一评估协议。
- FGSM/PGD 是基线，后续研究应进一步加入 4D 雷达物理约束、稀疏性约束、目标区域攻击和跨模型迁移实验。
