# 4D Radar 对象级对抗攻击研究阶段报告

更新日期：2026-08-06
代码分支：`research/4d-radar-attacks`
当前阶段：完成基础攻击、I-ADV-RCS 直接迁移、对象级损失和梯度诊断；尚未形成最终 Radar-aware 新方法

## 1. 报告目的

本文总结目前在 View-of-Delft（VoD）4D Radar 点云检测上的对抗攻击研究，包括：

- 当前课题定义和威胁模型；
- 已实现、已运行和仅作为基础设施保留的方法；
- 数据统计、100 帧筛选、完整验证集 AP 和对象级诊断结果；
- 当前实验能够支持的结论，以及不能支持的结论；
- I-ADV 迁移效果差、简单对象损失效果受限的原因；
- 下一阶段可能形成论文方法的研究路线。

本报告特别区分三种状态：

1. **已实现且有正式实验结果**：可以作为当前阶段结论依据；
2. **已实现但只做过 smoke test**：只能说明代码可运行；
3. **研究建议或方法假设**：尚未实现，不能写成已有贡献。

## 2. 当前课题定义

当前课题不应只定义为“RCS 攻击”，更合适的表述是：

> **4D Radar 点云检测器的对象级白盒对抗攻击研究。**

RCS-only I-ADV-RCS 是 LiDAR 方法迁移基线，不是最终课题边界。4D Radar 点云同时包含几何、反射、速度和时间信息，因此真正具有研究价值的问题是：

> 如何根据不同目标和不同 Radar 特征的敏感度，在受约束预算下联合破坏目标的分类证据与定位质量？

## 3. 当前威胁模型

| 维度 | 当前定义 |
| --- | --- |
| 攻击知识 | 白盒，可访问模型参数、训练损失、输出和输入梯度 |
| 攻击阶段 | 数字域，直接修改输入检测器的 4D Radar 点云 |
| 攻击范围 | 当前主实验为对象级，同时攻击场景中满足条件的 Car |
| 攻击目标 | 类别条件的非定向消失攻击；不指定错误类别 |
| 点集合 | 只修改已有点，不增加、删除或重新排序点 |
| 输入时序 | 使用 VoD 五帧累积点云，不是只使用 `time=0` 当前帧 |
| 目标选择 | 主要使用 clean 中按 VoD 标准检测成功的 Car |
| 主要匹配标准 | Car `3D IoU >= 0.5` |
| 当前正式扰动属性 | 单独的 XYZ、RCS 或 Doppler；尚未完成多模态联合方法 |
| 当前模型 | VoD 五帧 Radar PointPillars，`AnchorHeadSingle` |

需要注意：对象级攻击使用 GT 框确定可修改点，因此属于较强的白盒监督设定。它适合方法探索和上界研究，但不能直接宣称现实攻击者天然知道准确 GT 框。

## 4. 数据、模型和评估设置

### 4.1 数据和模型

- 数据集：VoD 五帧累积 Radar 点云；
- 验证集：1296 帧；
- 输入特征：`[x, y, z, rcs, v_r, v_r_comp, time]`；
- 模型：`cfgs/kitti_models/pointpillar_radar.yaml`；
- checkpoint：`vod5_retrain_seed0/checkpoint_epoch_80.pth`；
- 模型 pillar 尺寸：`[0.16, 0.16, 5] m`；
- 默认随机种子：1024。

I-ADV 的攻击分组尺寸是攻击算法自身的立方体尺寸，不等于模型 pillar 尺寸。

### 4.2 100 帧筛选协议

大多数方法探索实验使用从 1296 帧中均匀抽取的相同 100 帧：

- `sample_strategy=uniform`；
- `batch_size=1`；
- PGD/I-ADV 均为 10 步；
- object-level 实验只修改 Car GT 框内、实际进入模型的点；
- fixed 模式保持 clean pillar 拓扑，避免将离散换柱与连续梯度混在一起。

### 4.3 当前指标

目前同时使用以下指标：

- Recall 和 Recall drop；
- sample-level ASR：较粗，只用于历史结果；
- Object ASR：clean 检测成功的 Car 在攻击后跌破 `3D IoU=0.5` 的比例；
- 连续对象终点指标：IoU drop、matched-score drop、对象证据下降、中心误差变化；
- VoD 官方 entire-area 和 ROI 3D/BEV AP、AOS。

Object ASR 是阈值指标。若目标 clean IoU 离 0.5 很远，即使 IoU 明显下降也不会记为成功，因此方法探索阶段不能只看 ASR。

## 5. 已完成的工程基础

### 5.1 独立 Radar 攻击模块

当前实现位于 `tools/radar_attack`，不依赖旧的 `tools/attack`，但依赖 OpenPCDet 的数据、模型、体素化和训练损失。

已具备：

- 点云 FGSM/PGD；
- 体素 FGSM/PGD 基线；
- I-ADV-RCS；
- 可微 PointPillars 体素适配；
- fixed/revoxelize 两种点云攻击模式；
- 对抗点云保存；
- 可恢复 YAML campaign；
- VoD 官方 AP 适配；
- Object ASR 和连续对象终点指标；
- 特征统计、对象敏感度和混合梯度诊断。

### 5.2 当前测试状态

- 38 个 `radar_attack` 单元测试通过；
- 已使用真实 checkpoint 完成 FGSM、PGD、I-ADV、对象证据损失和混合损失 GPU smoke test；
- 已验证最大扰动不超过 epsilon；
- I-ADV-RCS 中 xyz、Doppler、time、点数和点顺序保持不变；
- object 邻域模式跨目标邻居数严格为 0；
- 最近实现尚未提交，当前工作树中还包含待审核的 I-ADV、对象损失、诊断和报告修改。

## 6. 当前使用过的方法

### 6.1 点云 FGSM 和 PGD

攻击发生在硬体素化之前的原始点云上。可分别修改：

- `xyz`；
- `rcs`；
- `doppler`，即当前模型使用的 `v_r` 和 `v_r_comp`；
- 也支持统一特征组合，但尚未作为正式 Radar-aware 方法研究。

FGSM 使用一次 sign 梯度更新，PGD 在 clean 点云的 L-infinity 预算内迭代更新。历史全场景基线使用 `revoxelize`，对象级方法比较使用 `fixed`，两类结果不能直接混为同一种攻击协议。

### 6.2 体素 FGSM 和 PGD

这是最早已有的基线，直接攻击模型体素特征。它生成的是对抗体素，不是可独立输入其他模型的对抗点云，因此不适合作为当前“对象级对抗点云”主方法。

体素攻击仍有两个用途：

- 作为模型内部最容易攻击的参考上界；
- 帮助区分“原始点云约束”与“网络中间特征脆弱性”。

最近实验重点已转移到点云攻击，没有继续扩充体素方法。

### 6.3 I-ADV-RCS 直接迁移

I-ADV 原本面向 LiDAR intensity。当前迁移将 intensity 映射为 Radar RCS，保留其主要结构：

1. 对目标点建立 KD-tree 邻域；
2. PCA 拟合局部表面法向量；
3. 结合法向角和距离构造反射特征；
4. 对 RCS 梯度执行 L1 归一化和 MI-FGSM 动量；
5. 使用反射特征增强梯度；
6. 在攻击立方体内使用正负极值比较决定统一更新方向；
7. 投影回 clean RCS 的 epsilon 邻域。

当前默认参数：

| 参数 | 值 |
| --- | ---: |
| 迭代次数 | 10 |
| `mu` | 1.0 |
| `lambda` | 1000 |
| `d_max` | 75 m |
| k 近邻 | 16，包含查询点 |
| PCA 最小点数 | 3 |
| 攻击分组边长 | 0.1 m |
| 邻域范围 | object |

当前实现还修正了第一版中“不同车辆共享 KD-tree 邻域和融合分组”的问题。默认按 `(batch_id, object_id)` 隔离，重叠 GT 框中的点分配给归一化中心距离最近的目标。

### 6.4 对象证据损失 `object_evidence`

训练损失同时包含全局分类、定位和方向损失，不直接对应“让某个 clean 已检出目标消失”。因此实现了对象证据损失：

1. 在 clean 输入上筛选检测成功目标；
2. 固定目标附近的 pre-NMS anchor 候选；
3. 使用目标类别 logits 的 LogSumExp 作为对象证据；
4. 攻击最大化负对象证据，从而同时压低一组候选，而不是只压低一个框。

候选 identities 在迭代过程中保持不变，避免攻击通过不断切换候选逃避优化。

### 6.5 分类—定位混合损失 `object_hybrid`

为了同时利用对象证据损失的降分能力和训练损失的定位破坏能力，进一步实现：

```math
L_{hybrid} = \frac{1}{N}\sum_o
\left[-E_o + \beta L_{loc,o}\right]
```

其中：

- `-E_o` 为负 LogSumExp 对象证据；
- 定位项从 clean decoded-IoU 最高的固定 top-32 anchors 中计算；
- 使用 clean 类别置信度作为固定权重；
- 增大当前 box encoding 和 GT encoding 的 Smooth L1 误差；
- yaw 使用正弦周期差；
- 先逐目标计算，再跨目标平均。

当前只实验了固定 `beta=1`。该设置已经证明无效，但不等于“联合分类和定位”的方向无效。

### 6.6 对象级梯度诊断

诊断器目前能够测量：

- training loss 与 object evidence loss 对目标证据的局部作用；
- IQR-normalized sensitivity；
- actual probe-budget-normalized sensitivity；
- geometry、Doppler、RCS 的对象级预算比例；
- 敏感度与距离、点数、速度和类别的关系；
- 分类与定位梯度的 L1 范数、平衡 beta、余弦相似度和符号冲突。

该诊断器不生成正式对抗样本，而是为新方法设计提供依据。

## 7. 输入特征统计

完整 1296 帧中共有 1,254,408 个处理后原始点，其中 964,238 个点作为硬体素有效条目进入统计，占 76.87%。

| 特征 | IQR | 备注 |
| --- | ---: | --- |
| x | 19.032 m | 受前向距离范围影响 |
| y | 6.775 m |  |
| z | 1.037 m |  |
| RCS | 15.590 | 当前 Radar RCS 预算依据 |
| `v_r` | 3.235 m/s | 原始径向速度 |
| `v_r_comp` | 0.0151 m/s | 分布高度集中在 0 附近 |
| time | 2 | 离散 sweep 编号，不是连续秒数 |

因此 RCS 预算筛选使用：

- `epsilon_rcs=0.8`，约 5% IQR；
- `epsilon_rcs=1.6`，约 10% IQR；
- `epsilon_rcs=3.1`，约 20% IQR。

time 的取值是 `{-4,-3,-2,-1,0}`。对它施加连续 FGSM/PGD 不具有清楚的物理意义，因此当前没有把 time 作为正式攻击变量。

## 8. 实验结果

### 8.1 全场景点云 FGSM/PGD 基线

历史基线在相同 100 帧上修改全场景点，使用训练损失和 `revoxelize`。

#### XYZ

| 方法 | epsilon | Recall drop |
| --- | ---: | ---: |
| FGSM | 0.01 | 0.0150 |
| FGSM | 0.02 | 0.0205 |
| FGSM | 0.04 | 0.0327 |
| PGD | 0.01 | 0.0191 |
| PGD | 0.02 | 0.0246 |
| PGD | 0.04 | **0.0505** |

#### RCS

| 方法 | epsilon | Recall drop |
| --- | ---: | ---: |
| FGSM | 0.1 | 0.0000 |
| FGSM | 0.3 | 0.0027 |
| FGSM | 0.6 | 0.0068 |
| PGD | 0.1 | 0.0000 |
| PGD | 0.3 | 0.0041 |
| PGD | 0.6 | **0.0096** |

#### Doppler

| 方法 | epsilon | Recall drop |
| --- | ---: | ---: |
| FGSM | 0.02 | 0.0000 |
| FGSM | 0.05 | 0.0055 |
| FGSM | 0.10 | 0.0150 |
| PGD | 0.02 | 0.0000 |
| PGD | 0.05 | 0.0082 |
| PGD | 0.10 | **0.0218** |

这组结果说明模型对 XYZ 最敏感，其次是 Doppler，RCS-only 在小预算下较弱。但不同特征的数值单位、物理代价和 epsilon 不同，不能只按 Recall drop 排名就声称某种物理攻击更强。此外，该实验是全场景 `revoxelize`，不能与后续对象级 fixed 实验直接比较。

### 8.2 I-ADV 逐目标隔离一致性实验

直接使用 LiDAR 风格 `0.1 m` 分组和 object 邻域时：

| RCS epsilon | Object ASR | 成功数/clean 已检出 |
| ---: | ---: | ---: |
| 0.1 | 0 | 0/133 |
| 0.3 | 0 | 0/133 |
| 0.6 | 0 | 0/133 |

100 帧中的 I-ADV 稀疏性诊断为：

- 261 个包含有效 Radar 点的 Car 目标；
- 平均每目标 18.67 点；
- PCA fallback 率 0.78%；
- `0.1 m` 分组中 91.27% 是单点组；
- 平均每组只有 1.117 点；
- 跨目标邻居数为 0；
- 非有限梯度步数为 0。

这说明实现本身稳定、对象隔离正确，主要问题不是 PCA 大量失败，而是 Radar 太稀疏，使 I-ADV 的“组内梯度融合”在绝大多数点上退化为逐点符号更新。

### 8.3 RCS 预算筛选

固定 10 步、object 范围和 `0.1 m` 分组：

| 方法 | epsilon | Object ASR | 成功数/133 | Recall drop |
| --- | ---: | ---: | ---: | ---: |
| PGD | 0.8 | 0 | 0 | 0.0000 |
| I-ADV | 0.8 | 0 | 0 | 0.0000 |
| PGD | 1.6 | **2.26%** | **3** | 0.0041 |
| I-ADV | 1.6 | 1.50% | 2 | 0.0000 |
| PGD | 3.1 | **5.26%** | **7** | 0.0082 |
| I-ADV | 3.1 | 2.26% | 3 | -0.0014 |

直接迁移的 I-ADV-RCS 在三个预算上都没有超过普通 object-level RCS-PGD。`epsilon=3.1` 时 I-ADV 的整体 Recall 甚至轻微上升，说明它的更新方向没有稳定对齐“让目标消失”的评价目标。

### 8.4 I-ADV 攻击分组尺度

固定 `epsilon_rcs=1.6`：

| 分组边长 | 单点组比例 | 平均每组点数 | Object ASR |
| ---: | ---: | ---: | ---: |
| 0.1 m | 91.27% | 1.117 | 2/133 |
| 0.3 m | 71.22% | 1.505 | 2/133 |
| 0.5 m | 57.92% | 1.930 | 2/133 |
| 1.0 m | 39.49% | 2.970 | 2/133 |

增大分组确实缓解了稀疏性，但没有提高 Object ASR。这意味着 I-ADV 效果差不能只归因于 `0.1 m` 分组太小；其反射增强、极值融合和训练损失方向本身也可能不适合 Radar。

`0.3/0.5/1.0 m` 是 Radar 尺度消融，不能称为未经修改的原始 I-ADV 复现。

### 8.5 完整 1296 帧 VoD 官方 AP

完整验证集在 `epsilon_rcs=1.6` 下比较 object-level PGD 和 I-ADV。PGD 使用固定随机种子下的 random start，I-ADV 按原算法不使用 random start；因此这是实际基线比较，不是完全相同初始化的纯模块消融。

#### 对象指标

| 方法 | Object ASR | 成功数/clean 已检出 | Recall drop |
| --- | ---: | ---: | ---: |
| PGD | **2.744%** | **48/1749** | 0.00412 |
| I-ADV-RCS | 1.258% | 22/1749 | 0.00063 |

#### 官方 3D AP 绝对下降

| 方法 | Entire Car AP drop | Entire mAP drop | ROI Car AP drop | ROI mAP drop |
| --- | ---: | ---: | ---: | ---: |
| PGD | **1.3064** | **0.4538** | **0.5256** | **0.1942** |
| I-ADV-RCS | -0.0157 | -0.0049 | 0.0760 | 0.0414 |

负下降表示攻击后的数值轻微上升。I-ADV 的 entire-area 3D AP 变化接近零，ROI 也只出现很小下降；普通 RCS-PGD 明显更强。

因此当前可以明确写出：

> LiDAR I-ADV 直接迁移到 VoD Radar RCS 后没有表现出相对 PGD 的优势，说明 LiDAR intensity 的表面反射假设和梯度融合策略不能直接视为 Radar-aware 设计。

不能写成“I-ADV 已被完整复现到作者论文水平”，因为作者未公开代码，且 KD-tree 邻居数、退化处理、目标点提取等细节需要透明假设。

### 8.6 对象证据损失的局部诊断

在 100 帧、Car/Pedestrian/Cyclist 共 319 个可攻击 clean 已检出目标上，一步局部 probe 得到：

- 对象证据损失比训练损失产生更大证据下降的目标比例：96.24%；
- training loss 平均证据下降：0.0153；
- object evidence loss 平均证据下降：0.0558；
- object evidence 的局部证据下降约为 training loss 的 3.65 倍。

按照实际 capped probe budget 归一化后，平均敏感度分配为：

| 模态 | 平均分配 |
| --- | ---: |
| Geometry | 70.09% |
| RCS | 21.31% |
| Doppler | 8.60% |

这不是正式攻击预算，而是局部一阶敏感度。它说明只研究 RCS 会主动放弃当前模型最敏感的几何方向，同时不同目标的比例并不完全相同。

敏感度总量与目标点数高度相关：geometry、Doppler、RCS 对点数的 Spearman rho 分别约为 0.75、0.69、0.68。敏感度随距离增加而下降，其中 geometry 与距离的 rho 约为 -0.51，RCS 约为 -0.29。

因此不同点数和距离的目标不应天然共享完全相同的攻击预算。

### 8.7 training loss 与 object evidence 的 10 步攻击

在相同 clean-detected Car、相同点掩码、相同预算和 fixed topology 下：

#### RCS，`epsilon=1.6`

| 损失 | Object ASR | 平均 IoU drop | IoU 下降比例 | 分数 drop | 证据 drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| training | 3/133 | **0.01994** | **75.19%** | 0.02024 | 0.11815 |
| object evidence | 3/133 | 0.01451 | 50.38% | **0.03207** | **0.28111** |

#### XYZ，`epsilon=0.02`

| 损失 | Object ASR | 平均 IoU drop | IoU 下降比例 | 分数 drop | 证据 drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| training | 0/133 | **0.00257** | **78.95%** | 0.00454 | 0.02639 |
| object evidence | 0/133 | 0.00072 | 51.13% | **0.00809** | **0.08012** |

结论是：

- training loss 更擅长破坏定位和 IoU；
- object evidence 更稳定地压低分类分数和候选证据；
- 两者在当前预算下最终跨过 ASR 阈值的目标数相同或都为零；
- 分类和定位目标具有互补性，但也可能发生梯度冲突。

早期 `vod_object_endpoint_ablation` 的 score/center 匹配曾在 adv 最大 IoU 等于 0 时关联到远处无关同类框。该问题已修复，本报告使用后续 `vod_object_hybrid_ablation` 中重新计算的终点数据。

### 8.8 固定 beta 的混合损失

#### RCS

| 损失 | Object ASR | IoU drop | 分数 drop | 证据 drop |
| --- | ---: | ---: | ---: | ---: |
| training | 3/133 | **0.01994** | 0.02024 | 0.11815 |
| evidence | 3/133 | 0.01451 | 0.03207 | **0.28111** |
| hybrid，`beta=1` | 3/133 | 0.01458 | **0.03210** | 0.28101 |

#### XYZ

| 损失 | Object ASR | IoU drop | 分数 drop | 证据 drop |
| --- | ---: | ---: | ---: | ---: |
| training | 0/133 | **0.00257** | 0.00454 | 0.02639 |
| evidence | 0/133 | 0.00072 | 0.00809 | **0.08012** |
| hybrid，`beta=1` | 0/133 | 0.00075 | **0.00809** | 0.08009 |

RCS 中 91/133 个目标的 evidence 和 hybrid 得到完全相同的 IoU 变化，三种 loss 攻击成功的也是完全相同的三辆车。固定 `beta=1` 的定位项几乎没有改变 sign-PGD 更新方向。

### 8.9 分类—定位梯度尺度和冲突测量

在同一 100 帧的 132 个可攻击 clean 已检出 Car 上，定义：

```math
\beta_o = \frac{\lVert g_{cls,o}\rVert_1}
                 {\lVert g_{loc,o}\rVert_1 + \delta}
```

测量结果：

| 模态 | 定位/分类梯度比中位数 | 平衡 beta 中位数 | beta 的 10%–90% | `beta=1` 改变 sign 比例 | 平衡后改变 sign 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XYZ | 0.00523 | 191 | 71–543 | 0.19% | 23.38% |
| Doppler | 0.00487 | 206 | 89–577 | 0.27% | 23.96% |
| RCS | 0.00418 | 239 | 82–662 | 0.20% | 25.17% |

RCS 的逐目标 beta 最小约 18、最大约 1533，说明一个固定全局 beta 很难适配所有车辆。

梯度方向也存在冲突：

| 模态 | 余弦中位数 | 余弦为负的目标比例 | 符号一致率中位数 |
| --- | ---: | ---: | ---: |
| XYZ | 0.123 | 40.9% | 54.6% |
| Doppler | 0.128 | 38.6% | 53.8% |
| RCS | 0.196 | 38.6% | 53.5% |

RCS beta 与 clean IoU 显著正相关，`rho=0.333`；与距离显著负相关，`rho=-0.252`；与点数的相关性较弱且不显著。

由此确认：

1. `beta=1` 失败的首要原因是定位梯度比分类梯度小约两个到三个数量级；
2. 即使完成范数平衡，约 39%–41% 的目标仍存在整体梯度冲突；
3. 下一步需要逐目标梯度平衡和冲突处理，而不是盲目枚举固定 beta。

## 9. 当前方法存在的主要问题

### 9.1 I-ADV 的 LiDAR 假设不能直接用于 Radar

LiDAR intensity 与 Radar RCS 不是同一种物理量：

- LiDAR intensity 更接近光学/近红外反射响应；
- Radar RCS 由电磁散射、目标几何、材料、姿态、频率和多径共同决定；
- I-ADV 的 `sin(angle) * sin(distance)` 反射增强在 Radar 上没有得到物理验证；
- VoD RCS 是否已经包含距离补偿也尚未在当前工作中确认。

因此当前 I-ADV-RCS 只能称为算法结构迁移基线，不能称为物理可实现 Radar 攻击。

### 9.2 Radar 稀疏性削弱分组融合

`0.1 m` 分组中约 91% 是单点组，极值融合几乎失去意义。增大到 `1.0 m` 虽然将单点组比例降到约 39%，但 ASR 没有提高，说明简单扩大分组不是充分解决方案。

### 9.3 五帧累积的邻域语义尚未解决

当前模型使用五帧累积点云。object 邻域会隔离不同车辆，但同一车辆的 KD-tree 邻居仍可能来自不同 sweep。尚未分析：

- 邻居的 time 构成；
- 运动补偿误差对 PCA 法向的影响；
- 当前帧点和历史帧点是否应使用不同预算；
- 只改当前 sweep 与同时改五帧的攻击能力差异。

因此当前攻击并不是“单帧模型专用”，但也还不是时间感知攻击。

### 9.4 数字预算不等于物理预算

- `epsilon_rcs` 目前按数据 IQR 定义，只是数字域合理尺度；
- XYZ 独立 Cartesian 扰动不直接对应 Radar 的 range/azimuth/elevation 测量误差；
- `v_r` 和 `v_r_comp` 在物理上存在关系，当前独立修改只能作为模型输入域基线；
- 尚未建立不同特征之间统一的物理成本。

所以多模态方法必须区分“模型归一化敏感度”和“物理可实现约束”。

### 9.5 当前 loss 仍是后处理代理目标

- object evidence 优化 pre-NMS 类别 logits，不直接优化最终 NMS 后 IoU；
- localization 使用 encoded Smooth L1，不等价于直接降低最终 3D IoU；
- 固定候选提升了迭代稳定性，但攻击后可能由其他候选框接管；
- 同场景多个目标通过共享网络特征产生交叉梯度，当前只做逐目标自身诊断，尚未处理目标之间的冲突。

### 9.6 Object ASR 太离散

100 帧中 clean 已检出的 133 个 Car，clean IoU margin 中位数约为 0.205，只有少量目标靠近 0.5 阈值。当前平均 IoU drop 约 0.015–0.020，因此不同 loss 很容易得到相同 ASR。

后续筛选必须同时报告连续终点指标，正式实验再以 Object ASR 和 AP 为主。

### 9.7 泛化和统计可靠性不足

目前只有：

- 一个数据集；
- 一个五帧 PointPillars checkpoint；
- 主要是一个随机种子；
- 没有跨模型白盒结果；
- 没有对抗点云迁移到其他模型的 black-box transfer 结果；
- 尚未重复实验估计均值和方差。

因此现在不能声称“对不同模型有效”或“具有跨架构泛化性”。点云格式可以输入其他模型，不等于在源模型上生成的对抗点云一定能迁移成功。

### 9.8 复现细节不完整

I-ADV 作者未公开代码，论文也没有明确给出所有实现细节。当前 `k=16`、L1 梯度归一化、PCA fallback、GT 框点提取等属于透明冻结的合理假设，必须在论文中说明。

## 10. 当前研究结论

目前可以较有把握地得出以下结论：

1. **4D Radar PointPillars 对原始点云扰动敏感，但不同模态差异明显。** 在当前数字预算下，XYZ 最敏感，RCS-only 较弱。
2. **LiDAR I-ADV 直接迁移没有优于普通 RCS-PGD。** 100 帧 Object ASR 和完整验证集官方 AP 均支持这一点。
3. **I-ADV 的问题不只是攻击分组尺寸。** 增大分组缓解稀疏性却没有提高 ASR。
4. **对象证据 loss 能更稳定地降低目标分数和候选证据。** 但训练 loss 更能破坏定位和 IoU。
5. **简单固定权重混合 loss 无效。** `beta=1` 几乎等价于纯对象证据攻击。
6. **分类和定位梯度存在严重尺度失衡与对象差异。** 平衡 beta 的典型值约为 200，而不是 1，并且跨目标范围很大。
7. **分类和定位还存在方向冲突。** 单纯把定位项乘大仍可能牺牲分类攻击方向。
8. **只完善 RCS-PGD、MI-FGSM 等迁移基线不足以形成论文创新。** 当前最有潜力的研究问题是对象自适应、多模态、冲突感知的攻击方向和预算分配。

当前不能声称：

- 已经提出最终新方法；
- I-ADV-RCS 具有物理可实现性；
- 攻击能泛化到不同模型；
- 固定 beta 取 200 就一定有效；
- 局部敏感度比例可以直接作为物理预算比例。

## 11. 建议的下一阶段方法方向

下一阶段建议暂时称为工作假设，而不是正式命名方法：

> **对象自适应、冲突感知、多模态 4D Radar 点云攻击。**

### 11.1 逐目标分类—定位梯度平衡

对每个目标和每个模态分别计算：

```math
\hat g_{cls,o,d} =
\frac{g_{cls,o,d}}{\lVert g_{cls,o,d}\rVert_1 + \delta}
```

```math
\hat g_{loc,o,d} =
\frac{g_{loc,o,d}}{\lVert g_{loc,o,d}\rVert_1 + \delta}
```

这比直接学习或枚举固定 beta 更符合当前测量结果。需要保留 clip，防止极小定位梯度产生异常放大。

### 11.2 分类—定位冲突处理

若：

```math
\hat g_{cls}\cdot\hat g_{loc}<0
```

则不能直接相加。可以比较：

- 不处理冲突；
- 将定位梯度投影到与分类梯度不冲突的子空间；
- 将两项目标视为多目标优化问题，寻找共同下降/上升方向；
- 根据 clean IoU margin 动态决定更偏分类还是定位。

这一步可能是比“换一种 loss”更有研究价值的核心。

### 11.3 对象级模态预算分配

当前局部诊断显示 geometry/RCS/Doppler 的实际 probe-budget 敏感度平均约为 70%/21%/9%，且与距离、点数相关。可以研究：

- 按目标敏感度分配数字预算；
- 按 clean IoU margin、距离、点数和速度调节分配；
- 将模型敏感度除以模态物理成本，再分配预算；
- 对当前 sweep 与历史 sweeps 使用不同权重。

不能直接把 70/21/9 当成最终预算，因为 XYZ、RCS 和 Doppler 的物理代价还未统一。

### 11.4 Radar 坐标和时序约束

后续 Radar-aware 表达应优先考虑：

- 用 range/azimuth/elevation 代替独立 XYZ 作为物理坐标约束；
- 联合约束 `v_r` 与 `v_r_comp`；
- 区分 `time=0` 和历史 sweep；
- 分析跨 sweep 邻域对局部几何的影响；
- 研究固定点修改之外的帧删除、点隐藏或回波注入，但这些属于新的威胁模型，不能与当前结果混报。

## 12. 建议的实验顺序

### 阶段 A：验证自适应梯度方向

在相同 100 帧上比较：

1. classification only；
2. localization only；
3. naive hybrid，`beta=1`；
4. 固定全局 beta，例如诊断中位数附近；
5. object-wise L1-balanced hybrid；
6. object-wise balanced + conflict projection。

先只使用 RCS 和 XYZ 单模态，确认方法机制，而不是立即把所有特征混在一起。

验收指标：

- Object ASR；
- mean/median IoU drop；
- score/evidence drop；
- center error increase；
- 每轮梯度余弦、sign 变化比例；
- 非有限梯度、预算和非攻击特征不变量。

### 阶段 B：多模态预算

在阶段 A 最优损失上比较：

- RCS；
- XYZ；
- RCS + Doppler；
- XYZ + RCS；
- XYZ + RCS + Doppler；
- 固定均匀预算与对象自适应预算。

### 阶段 C：完整评估

只有当 100 帧连续指标和 Object ASR 都稳定优于 training PGD、object evidence 和 I-ADV-RCS 后，再运行：

- 完整 1296 帧 VoD 官方 AP；
- 至少 3 个随机种子或确定性重复；
- 不同距离、点数、速度和 clean IoU 分层结果。

### 阶段 D：跨模型和迁移

在主方法有效后：

- 增加至少一个不同检测头或表示方式的 4D Radar 检测器；
- 分别报告每个模型上的白盒攻击；
- 将源模型生成的对抗点云直接输入目标模型，报告 black-box transfer；
- 区分“输入格式通用”和“攻击效果可迁移”。

## 13. 当前研究位置判断

目前已经完成了论文研究中非常重要但仍属于前期的工作：

- 建立了独立、可复现、可运行官方评估的攻击框架；
- 完成了 LiDAR SOTA 方法 I-ADV 到 4D Radar 的透明迁移；
- 通过完整验证集证明直接迁移效果有限；
- 找到了普通训练 loss 与对象证据 loss 在定位和分类上的分工；
- 通过梯度测量解释了简单混合 loss 为什么失败；
- 得到了“对象自适应梯度平衡 + 冲突处理 + 多模态预算”这一有数据依据的方法方向。

但目前仍处于：

> **已经定位研究问题并形成方法假设，尚未完成最终方法验证。**

这不是走偏。相比继续机械复现 MI-FGSM、NI-FGSM 等基线，现在已经开始回答更关键的问题：为什么 LiDAR 方法在 Radar 上失效，以及 4D Radar 的对象差异、多模态特征和分类—定位冲突应如何进入攻击设计。

## 14. 结果和代码索引

| 内容 | 路径 |
| --- | --- |
| Radar 攻击说明 | `tools/radar_attack/README.md` |
| I-ADV 迁移表 | `tools/radar_attack/docs/iadv_to_4d_radar.md` |
| I-ADV 实现 | `tools/radar_attack/attacks/iadv.py` |
| 对象损失 | `tools/radar_attack/attacks/objective.py` |
| 对象诊断器 | `tools/radar_attack/diagnose_object_evidence.py` |
| 特征统计 | `output/radar_attack/pointpillar_radar/feature_statistics.json` |
| 全场景基线 | `output/kitti_models/pointpillar_radar/vod_point_baseline_screen/summary.md` |
| I-ADV 预算与尺度 | `output/kitti_models/pointpillar_radar/vod_iadv_rcs_radar_screen/summary.md` |
| 完整 VoD AP | `output/kitti_models/pointpillar_radar/vod_iadv_rcs_full_ap/summary.md` |
| 对象连续终点 | `output/kitti_models/pointpillar_radar/vod_object_hybrid_ablation/summary.md` |
| 对象敏感度 | `output/radar_attack/pointpillar_radar/object_evidence_dual_scale_100/summary.json` |
| 分类—定位梯度诊断 | `output/radar_attack/pointpillar_radar/hybrid_gradient_relationship_100/summary.json` |
