# RadarPillars（五帧 VoD）

这个接入与原有 `pointpillar_radar` 并存，不会修改它的配置、VFE、权重或输出目录。

## 文件和来源

- 模型配置：`tools/cfgs/kitti_models/radarpillars.yaml`
- 独立 VFE：`pcdet/models/backbones_3d/vfe/radarpillars_vfe.py`
- pillar 注意力：`pcdet/models/backbones_3d/pillar_attention.py`
- 本地预训练权重：`pretrained/radarpillars/radarpillar_vod_best_map52.56.pth`
- 权重来源：第三方复现 [fthbng77/RadarPillar v1.0](https://github.com/fthbng77/RadarPillar/releases/tag/v1.0)
- SHA-256：`beb92e6cf7f7831c0d1111b6e2a9cca618ef8b3534d580dc49d68c4871d0d0da`

该权重使用 VoD 的 `radar_5frames` 输入。每个点必须是七维：
`[x, y, z, rcs, v_r, v_r_comp, time]`。

## 测试预训练权重

在仓库的 `tools` 目录运行：

```bash
cd tools
python test.py \
  --cfg_file cfgs/kitti_models/radarpillars.yaml \
  --ckpt ../pretrained/radarpillars/radarpillar_vod_best_map52.56.pth \
  --extra_tag pretrained_v1
```

结果写入 `output/kitti_models/radarpillars/pretrained_v1/`，不会进入
`output/kitti_models/pointpillar_radar/`。

数据仍沿用仓库已有的 KITTI 兼容目录，但新配置使用独立的 `VoDRadarDataset`
适配器，以免误跑不适用于 VoD 的 KITTI 指标。`test.py` 会生成 `result.pkl`，
再运行已有脚本计算 VoD 官方指标：

```bash
python eval_vod.py \
  --result-pkl ../output/kitti_models/radarpillars/pretrained_v1/eval/epoch_56/val/default/result.pkl \
  --label-dir ../data/view_of_delft/radar_5frames/training/label_2 \
  --output-dir ../output/kitti_models/radarpillars/pretrained_v1/vod_predictions
```

## 继续训练或从预训练权重微调

从头训练：

```bash
cd tools
python train.py \
  --cfg_file cfgs/kitti_models/radarpillars.yaml \
  --extra_tag from_scratch
```

以预训练模型初始化并重新开始训练：

```bash
cd tools
python train.py \
  --cfg_file cfgs/kitti_models/radarpillars.yaml \
  --pretrained_model ../pretrained/radarpillars/radarpillar_vod_best_map52.56.pth \
  --extra_tag finetune_v1
```

`--pretrained_model` 只加载模型参数，不恢复第三方检查点中的 epoch 和优化器状态。
若要恢复本仓库自己保存的训练断点，再使用 `--ckpt`。

## 与现有 PointPillars-radar 的隔离

- `Radar7PillarVFE` 保持不变；RadarPillars 使用新名字 `RadarPillarsVFE`。
- `pointpillar_radar.yaml` 保持不变。
- RadarPillars 的配置文件名决定其独立的输出路径。
- 预训练 `.pth` 继续受仓库的 `*.pth` 规则忽略，不会被意外提交。
