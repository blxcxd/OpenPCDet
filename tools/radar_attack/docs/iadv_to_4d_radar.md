# I-ADV 到 4D Radar RCS 攻击迁移表

状态：第一版透明复现、逐目标隔离、100 帧筛选和完整 VoD AP 已完成；直接迁移当前未优于同预算 RCS-PGD

更新日期：2026-08-06

最新完整结果与后续方法方向见
[`research_status_report_2026-08-06.md`](research_status_report_2026-08-06.md)。

## 1. 文档目的

本文档用于指导将 LiDAR intensity 攻击 I-ADV 迁移到 VoD 4D Radar RCS 点云攻击。它不是实现完成声明，也不是论文方法描述。

迁移分成两个严格区分的阶段：

1. **直接迁移基线（I-ADV-RCS）**：尽量保留 I-ADV 的算法结构，只做数据和检测框架所必需的映射。
2. **Radar-aware 改进**：在直接迁移结果清楚后，再引入 RCS、Doppler、距离、角度和时间等 Radar 特性。

不能在第一版中混入未经标注的 Radar 改进，否则无法判断性能变化来自原方法还是新设计。

当前实现位于 `tools/radar_attack/attacks/iadv.py`，命令行入口为 `--attack_type iadv`。实现没有声称补全作者未公开的代码；所有欠明确选择均作为可配置复现假设记录在命令行参数、结果 JSON 和实验 YAML 中。

## 2. 参考资料

- 最终论文：Junqi Wu et al., *Invisibility stickers against LiDAR: Adversarial attacks on point cloud intensity for LiDAR-based object detection*, Computer Vision and Image Understanding, 2026. DOI: [10.1016/j.cviu.2026.104812](https://doi.org/10.1016/j.cviu.2026.104812)
- 可检索的方法预印本：[OpenReview 哈希 PDF](https://openreview.net/pdf/9f8fe83091fa53e7024cda400b33fbff6f50d647.pdf)
- OpenReview 记录：[ICLR 2025 withdrawn submission](https://openreview.net/forum?id=P2snmtUBkQ)
- 实验参数依据：最终论文第 4.1 节 *Experimental setup*（已取得原文摘录）
- 当前 4D Radar 攻击入口：`tools/radar_attack/run_attack.py`
- 当前点云梯度攻击：`tools/radar_attack/attacks/gradient.py`
- 当前 OpenPCDet 适配器：`tools/radar_attack/adapters/openpcdet.py`

截至 2026-08-01，最终论文页面、OpenReview 记录、Crossref、Semantic Scholar、OpenAlex 和 GitHub 公开仓库搜索均未发现作者公开的代码仓库。OpenReview 投稿记录还明确填写了没有匿名代码 URL。当前能访问的附录只补充了 nuScenes 评估，没有补充实现参数。因此，后文将“论文明确写出”“根据 MI-FGSM 的合理推断”和“为复现实验自行冻结的假设”严格分开。

## 3. 当前威胁模型

第一阶段采用以下威胁模型：

| 维度 | 当前定义 |
| --- | --- |
| 攻击知识 | 白盒：可访问模型参数、损失和输入梯度 |
| 攻击域 | 数字域：直接修改生成后的 4D Radar 点云 |
| 攻击目标 | 非定向：降低整体检测性能，不指定目标类别或目标框 |
| 点集合 | 固定点数，只修改已有点，不增加或删除点 |
| 被攻击属性 | 第一版只修改 RCS |
| 坐标 | `x, y, z` 保持不变 |
| 速度 | `v_r, v_r_comp` 保持不变 |
| 时间 | `time` 保持不变 |
| 攻击位置 | 体素化之前的原始点云 |
| 约束 | RCS 分组预算和有效数值范围，具体定义待确认 |

I-ADV 原论文的优化目标是让检测输出发生错误，同时约束输入距离：

```math
\mathcal{F}(X) \ne \mathcal{F}(X^*),
\qquad \mathcal{D}(X, X^*) < \eta
```

算法输入包含网络权重 `theta` 和 ground-truth `y_gt`，因此属于白盒监督攻击。它没有要求模型输出某个指定错误类别，所以在输出目标意义上是非定向攻击；但实验以目标 Car 实例为样本，因此攻击范围又是 object-conditioned。本文分别使用 `scene-untargeted` 和 `car-object-hiding` 标记这两种实验协议。

VoD 当前使用的逻辑点特征为：

```text
[x, y, z, rcs, v_r, v_r_comp, time]
```

OpenPCDet 合批后的点张量会在最前面增加 `batch_idx`。实现时不得把固定列号写死为原始点云布局，应继续使用数据集的 `used_feature_list` 解析语义列。

## 4. 迁移标记

| 标记 | 含义 |
| --- | --- |
| `KEEP` | 原方法结构和定义直接保留 |
| `MAP` | 做语义对应替换，例如 LiDAR intensity 映射到 Radar RCS |
| `ADAPT` | 数据或框架差异导致必须适配；需要记录理由 |
| `DEFER` | 超出当前威胁模型，暂不实现 |
| `TODO` | 尚未从论文或代码中确认，禁止凭经验猜测 |

## 5. 模块迁移表

| 原方法模块 | LiDAR 原始定义 | 类型 | 4D Radar 直接迁移 | Radar 特有风险或疑问 | 验证方法 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 威胁模型 | 输入含网络权重和 `y_gt`；白盒、非指定错误输出、只修改 intensity | `KEEP/MAP` | 白盒、数字域、非定向，只修改 RCS | 原实验按目标 Car 统计，当前场景级协议需单独标记 | 检查梯度来源和攻击区域 | 原定义已确认 |
| 输入属性 | 坐标与 intensity 组成点 | `MAP` | 将 intensity 替换为 RCS | LiDAR intensity 与 Radar RCS 不是同一种物理量 | 只允许 RCS 列发生变化 | 待实现 |
| 点数量 | 不添加或删除点 | `KEEP` | 保持点数和点顺序 | 无 | 比较 clean/adv 点数和索引 | 待实现 |
| 空间坐标 | intensity 攻击不修改 xyz | `KEEP` | xyz 完全固定 | 无 | `max(abs(xyz_adv - xyz)) == 0` | 待实现 |
| 体素分区 | 每次迭代按 xyz 划分边长 `0.1 m` 的立方体，同一体素统一更新方向 | `KEEP` | 第一版使用独立的 `0.1 m` 立方体对 Radar 点分组 | xyz 不变，因此每轮分组理论上相同；Radar 稀疏时可能大量单点分组 | 统计单点体素比例并验证预计算分组与逐轮分组一致 | 原流程已确认 |
| 邻域搜索 | KD-tree 搜索每个点的 3D 邻居 | `KEEP` | 在 Radar xyz 上搜索邻域 | 累积帧中的邻居可能来自不同时刻；稀疏区域邻居不稳定 | 统计有效邻居数、距离和时间组成 | `TODO` 邻居数 |
| 局部平面 | 使用 PCA 和特征分解估计表面法向量 | `KEEP` | 第一版忠实使用 PCA | Radar 点难以形成稠密物体表面；法向量可能退化 | 检查特征值比、有限性和重复运行稳定性 | 待实现 |
| 角度特征 | 计算局部法向量与传感器射线的夹角 `phi` | `MAP` | 计算法向量与 Radar 原点到点的视线夹角 | 需要复核论文的角度方向和法向量符号约定 | 可视化 `phi` 分布并做边界样例 | `TODO` 角度约定 |
| 距离特征 | 使用点到 LiDAR 原点的欧氏距离 `d` | `MAP` | 使用点到 Radar 原点的欧氏距离 | VoD 的 RCS 是否已做距离补偿需要确认 | 统计 clean RCS 与距离的相关性和条件分布 | 待分析 |
| 梯度增强 | 循环前计算固定反射特征；每轮先累积归一化梯度动量，再乘 `lambda * f` | `MAP` | 对 RCS 梯度使用相同顺序 | LiDAR 的光学反射规律不能直接解释 Radar RCS | 对比原始、动量和增强梯度分布 | 原流程已确认 |
| 动量更新 | `g_0=0`；`g_(t+1)=mu*g_t + normalized_grad`，`mu=1.0`；正文说明使用 MI-FGSM 框架 | `KEEP` | 对 RCS 梯度使用相同动量公式 | 公式分母只写通用范数，未标范数类型；L1 只能由 MI-FGSM 推断 | 与无动量版本做数值测试 | 公式与框架已确认，范数需冻结假设 |
| 梯度融合 | 比较分组内增强梯度最大正极值与最小负极值的绝对值 | `KEEP` | 每组选择统一的 RCS 更新方向 | 单点分组中融合没有作用 | 与平均融合、符号平均和逐点更新消融比较 | 待实现 |
| 属性更新 | 每轮按极值融合方向为同一体素 intensity 加上 step `epsilon`，重建后进入下一轮 | `MAP` | 同一分组内 RCS 同方向迭代更新 10 次 | 伪代码未显示合法范围裁剪或相对 clean 投影 | 单元测试同组更新一致且预算不越界 | 更新顺序已确认，投影待确认 |
| 扰动预算 | intensity 范围 `[0,1]`，每步 `0.2`，最大扰动 `1.0` | `ADAPT` | 使用 `epsilon_rcs`；第一版步长取 `0.2 * epsilon_rcs` | LiDAR 数值范围不能直接照搬到 Radar RCS | 检查最大扰动和裁剪后分布 | 原参数已确认，Radar 预算待定 |
| 攻击损失 | 计算 `grad_X L(F(theta, X*_t, y_gt))` 并归一化后进入动量 | `ADAPT` | 第一阶段与当前 RCS-PGD 使用相同 OpenPCDet 检测损失 | 原论文未在本段展开 `L` 的具体组成和梯度范数 | 固定损失比较 PGD 与 I-ADV-RCS | 总体形式已确认，细节待确认 |
| 攻击区域 | 主实验明确是 object-level，只扰动目标车辆；另有 full-scene 实验 | `ADAPT/TODO` | `gt_boxes` 使用 GT Car 框；重叠框中的点按归一化中心距离分给唯一目标；`scene` 修改全场景模型有效点 | 论文仍未说明目标车辆点由 GT 框、裁剪样本还是其他规则得到 | 记录每帧目标数、每车点数和点归属 | 已实现；GT 框仍是透明复现假设 |
| 对象隔离 | 论文复杂度和主实验描述以目标物体内点为单位 | `ADAPT` | 默认按 `(batch_id, object_id)` 分别建立 KD-tree、PCA 邻域和融合分组 | 第一版 Radar 实现曾把一帧所有目标点合并，稀疏 Radar 下会人为补足邻域 | 断言跨目标邻居数为 0；构造相同立方体坐标的双目标测试 | 已实现，旧行为由 `attack_union` 保留 |
| 物理反射材料 | 用表面材料改变 LiDAR intensity | `DEFER` | 当前不实现、不声称物理可实现 | Radar 材料和电磁散射机制完全不同 | 不适用 | 延后 |
| 评价指标 | KITTI 用逐 Car `ASR@IoU 0.7`；nuScenes 改用 1 m 中心距离 | `ADAPT` | VoD 主 Car ASR 使用官方 3D IoU `0.5`，同时报告官方 entire-area/ROI AP、分类别 AP、Recall 和扰动统计 | 数据集匹配标准不能机械照搬 | 在报告中明确数据集、距离类型和阈值 | 原定义已确认，待实现 |
| 数据集与模型 | KITTI；PointRCNN、IA-SSD、PointPillar、Voxel R-CNN、PV-RCNN、PDV；附录包含 nuScenes | `ADAPT` | VoD；第一阶段只使用 PointPillars Radar 即可验证迁移 | 单模型不能支持最终的跨架构泛化结论，但不妨碍初次迁移 | 方法有效后再加入不同表示的 Radar 检测器 | 已确认原设置，非全部必做 |
| 对比基线 | point-wise：随机噪声、PGD、MI-FGSM、NI-FGSM；voxel-wise：Random、Average、Voting | `ADAPT` | 原论文基线用于理解方法位置，不要求第一阶段全部复现 | 盲目补齐所有基线会扩大工作量，却不直接回答 Radar 迁移是否有效 | 最小比较只保留 RCS-PGD 和必要模块消融 | 已确认原设置，非全部必做 |
| ASR | 以 Car 实例为样本；clean 满足数据集匹配标准且攻击后不再满足时成功 | `ADAPT` | 实现 VoD 对齐的逐实例 Car `ASR@3D IoU 0.5`；可附加报告 KITTI-style `ASR@0.7` | 当前 sample ASR 和 target ASR 都不等价于逐 Car 匹配 ASR | 对 GT Car 分别计算 clean/adv 最大 3D IoU | 已确认定义 |

## 6. 原方法核心公式

### 6.1 反射特征

根据可访问预印本，I-ADV 为第 `i` 个点构造反射特征：

```math
f_i = \sin(\phi_i)\cdot
      \sin\left(\frac{d_i}{d_{\max}}\frac{\pi}{2}\right)
```

其中：

- `phi_i`：局部表面法向量与传感器射线之间的角度；角度约定需要复核。
- `d_i`：点到传感器原点的距离。
- `d_max`：传感器最大检测距离。

第一版直接迁移应保留公式结构，但只能称为“LiDAR 公式的直接迁移”，不能在没有证据时解释为 Radar RCS 的真实物理模型。

### 6.2 归一化梯度、动量与梯度增强

```math
g_0 = 0
```

每轮先计算检测损失对输入的梯度并进行归一化，再累积动量：

```math
g_{t+1} = \mu g_t +
\frac{\nabla_X\mathcal{L}(\mathcal{F}(\theta, X_t^*, y_{gt}))}
     {\left\|\nabla_X\mathcal{L}(\mathcal{F}(\theta, X_t^*, y_{gt}))\right\|}
```

随后使用预先计算的逐点反射特征增强动量梯度：

```math
\hat{g}_{t+1} = \lambda f \odot g_{t+1}
```

- `mu`：动量衰减因子，原论文设置为 `1.0`。
- `lambda`：梯度增强系数，原论文设置为 `1000`。
- `f`：由固定 xyz 预先计算，在迭代中不重新估计。
- `odot`：逐点乘法。
- 正文明确说迭代更新使用 MI-FGSM 框架，但公式分母只写了通用范数，没有标明范数类型。标准 MI-FGSM 使用 L1 归一化，因此第一版可将 L1 作为有依据的复现假设；它不是 I-ADV 论文直接给出的参数，必须在配置和实验记录中注明。
- 实现对 `X` 求梯度后，只提取 RCS/intensity 对应分量参与更新，其他特征必须保持不变。

### 6.3 极值梯度融合

对体素或分组 `V_m`：

```math
g_{m,+}=\left|\max(\hat{g}_{V_m})\right|
```

```math
g_{m,-}=\left|\min(\hat{g}_{V_m})\right|
```

更新方向：

```math
s_m =
\begin{cases}
+1, & g_{m,+}\ge g_{m,-}\\
-1, & g_{m,+}<g_{m,-}
\end{cases}
```

同一分组内所有被攻击属性使用统一方向 `s_m`。论文算法用 `epsilon` 表示每轮更新量，而威胁模型用 `eta` 表示总体扰动预算；两者不能在代码参数中混为同一个概念。论文实验执行 10 次迭代，每次 intensity 更新量为 `0.2`，最大扰动限制为 `1.0`。

### 6.4 已确认的原论文实验参数

| 参数 | I-ADV 原论文设置 | 4D Radar 直接迁移决定 |
| --- | --- | --- |
| 迭代次数 | 10 | `KEEP`：第一版同样使用 10 次 |
| intensity 范围 | `[0, 1]` | `ADAPT`：不能用于 RCS；使用 VoD RCS 原始数值域 |
| 单步更新量 | `0.2` | `ADAPT`：保留“总预算的 0.2 倍”这一比例，具体值随 `epsilon_rcs` 变化 |
| 最大扰动 | `1.0` | `ADAPT`：不能直接照搬，使用 Radar RCS 预算 |
| 攻击分组体素 | 边长 `0.1 m` 的立方体 | `KEEP`：第一版先使用 `0.1 m`，不得误用模型的 `[0.16, 0.16, 5]` pillar 尺寸 |
| 衰减因子 `mu` | `1.0` | `KEEP`：用于归一化输入梯度的跨轮动量累积 |
| 梯度增强 `lambda` | `1000` | `KEEP`：第一版保留，并记录增强前后的梯度数值范围 |
| 最大检测距离 `d_max` | `75 m` | `ADAPT/TODO`：原值已确认；VoD 当前点云范围为 `[0,-25.6,-3,51.2,25.6,2]`，最远裁剪角点约 `57.3 m`，需决定使用原值还是数据范围对应距离 |

原论文的单步更新量恰好是最大扰动限制的 `0.2` 倍。为避免与论文符号混淆，Radar 实现中继续使用 `epsilon_rcs` 表示总预算，使用 `step_size` 表示单步更新量：

```math
\alpha_{rcs} = 0.2\,\epsilon_{rcs}
```

这也等价于当前 10 步 PGD 默认使用的 `2 * epsilon / steps`，因而便于进行同预算公平比较。这里保留的是相对比例，不是把 LiDAR 的 `0.2` 数值直接复制到 RCS。

### 6.5 原论文算法流程

根据 Algorithm 1，直接迁移的执行顺序应为：

1. 从 clean xyz 计算每个点的角度 `phi`、距离 `d` 和固定反射特征 `f`。
2. 初始化 `g_0 = 0`、`X_0^* = X`。
3. 每轮对当前 `X_t^*` 做攻击体素分组。
4. 前向计算检测损失，并求损失对输入的归一化梯度。
5. 使用 `mu` 累积梯度动量。
6. 使用 `lambda * f` 增强动量梯度。
7. 在每个体素内比较最大正、负极值，得到统一更新方向。
8. 为该体素中的 RCS/intensity 加上单步更新量。
9. 重建点云并进入下一轮。
10. 返回第 `T` 轮对抗点云。

由于只修改属性、xyz 始终固定，角度、距离和体素成员关系在数学上不随迭代改变。实现可以缓存这些量以提高效率，但必须用测试证明缓存版本与按论文逐轮重新分组的结果一致。

论文给出的总体复杂度为 `O(T * M * N^2)`，其中 `N` 被描述为目标物体内的点数。论文后续还明确说明前面的主实验将扰动限制在目标车辆，即 object-level perturbation，并把 full-scene interference 作为另一项实验。由此可以确认需要支持目标物体掩码；但论文仍未说明这个掩码是按 GT 3D 框内点、预先裁剪样本还是其他规则生成。

### 6.6 KITTI 实验的 ASR 定义

原论文只以 Car 类别目标为评价对象，并将场景中的每个 Car 实例视为一个样本。对某个 GT Car：

```text
clean 最大 3D IoU > 0.7
并且
adversarial 最大 3D IoU <= 0.7
```

则该目标记为一次成功攻击：

```math
ASR = \frac{\text{攻击后检测失败的 clean 已检出 Car 数}}
           {\text{clean 已检出的 Car 总数}}
```

当前框架的 `attack_success_rate_sample` 只判断一帧是否还存在超过置信度阈值的预测；`attack_success_rate_target` 使用整体 Recall 计数差。两者都不是上面的逐实例匹配定义。因此实现 I-ADV-RCS 时需要新增论文对齐的 object-level ASR，不能直接用现有 ASR 声称复现论文结果。

只选择 Car 作为评价对象不等于强迫模型输出某个指定错误类别，但它把攻击范围收窄为特定类别/目标的消失攻击。后续必须分别标记：

- `scene-untargeted`：当前全场景非定向损失；
- `car-object-hiding`：论文对齐的 Car 目标消失协议。

两种协议的结果不能混在同一列直接比较。

### 6.7 nuScenes 附录与数据集匹配标准

论文附录 A 没有补充攻击实现细节，而是验证跨数据集和跨模型效果。它在 nuScenes val 的 Car 类别上使用 `1 m` 欧氏中心距离作为检测匹配标准，而不是沿用 KITTI 的 3D IoU `0.7`。I-ADV 在 PointPillar、CenterPoint-VoxelNet 和 VoxelNeXt 上分别报告 `84.5%`、`85.8%` 和 `89.0%` ASR。

## 7. 逐目标隔离与 Radar 筛选协议

当前实现将两个概念分开：

- `iadv_scope` 决定实际修改哪些点：`gt_boxes` 或 `scene`；
- `iadv_neighbor_scope` 决定 KD-tree/PCA 和融合的隔离范围：`object`、`attack_union` 或 `scene`。

论文主实验复现默认使用 `gt_boxes + object`。第一版未隔离实现对应 `gt_boxes + attack_union`，只作为直接迁移历史基线保留。论文全场景实验使用 `scene + scene`。`object` 模式中，跨目标邻居数必须严格为零。

VoD RCS 的测得 IQR 为 `15.59`。Radar 预算筛选采用 `epsilon_rcs={0.8,1.6,3.1}`，约为 IQR 的 `{5%,10%,20%}`，每组保持 10 步和 `step_size=0.2*epsilon_rcs`。分组尺度筛选固定 `epsilon_rcs=1.6`，仅改变攻击立方体边长 `{0.1,0.3,0.5,1.0} m`，不得把 Radar-aware 尺度结果称为未经修改的原始 I-ADV。

若 Object ASR 在不超过 10% IQR 的预算开始非零，再选择代表设置运行完整 1296 帧 VoD 官方 AP。若 `epsilon_rcs=3.1` 且分组边长 `1.0 m` 仍为零，则停止盲目增大预算，转向 full-scene、Doppler-RCS 联合攻击或 Radar-aware 检测损失。

这说明 I-ADV 的 ASR 核心定义是：

```text
clean 中按当前数据集标准成功匹配的目标
攻击后不再满足同一匹配标准
```

因此迁移到 VoD 时必须使用 VoD 的匹配标准。当前本地官方评估代码 `vod/evaluation/kitti_official_evaluate.py` 实际报告的第二组阈值为：

| 类别 | BEV/3D IoU 阈值 |
| --- | ---: |
| Car | 0.5 |
| Pedestrian | 0.25 |
| Cyclist | 0.25 |

I-ADV-RCS 第一阶段仍以 Car 为论文对齐对象，但主指标应是 VoD-style `Car ASR@3D IoU 0.5`。可以附加计算 `ASR@0.7` 观察与 KITTI 协议的差异，但不能用它代替 VoD 官方标准。

## 7. 两个版本的边界

### 7.1 直接迁移基线：I-ADV-RCS

第一版只允许包含以下必要变化：

```text
LiDAR intensity          -> VoD Radar RCS
LiDAR 数据读取           -> OpenPCDet/VoD 数据适配
LiDAR 检测模型接口       -> 当前 PointPillars 接口
原数据范围与预算         -> 明确记录的 Radar RCS 范围与预算
```

应尽量保留：

- 笛卡尔体素分区；
- KD-tree 邻域；
- PCA 法向量；
- 原角度—距离公式；
- 极值梯度融合；
- 固定点数和固定 xyz。

如果由于框架限制必须改变原算法，应在本文档的“决策记录”中单独登记。

### 7.2 后续方法：Radar-aware I-ADV

以下内容属于后续研究候选，不能悄悄加入直接迁移基线：

- range–azimuth–elevation 分区；
- 基于 Doppler 的梯度权重；
- 区分累积点的时间索引；
- 用 Radar 稀疏结构替代 PCA 表面法向量；
- 建模 RCS 与距离、角度、类别的条件关系；
- RCS 与 Doppler 联合攻击；
- 限制被修改点比例或只攻击目标相关点；
- 加入测量误差和物理可实现约束。

## 8. 分层比较协议

迁移 I-ADV 不等于复现它在 KITTI 上的完整实验。原论文的全部模型和基线记录在本文档中，是为了理解方法和避免遗漏，并非当前都要实现。

### 8.1 第一阶段：迁移能否运行且是否有初步价值（立即必做）

最小实验只比较当前直接基线和完整迁移方法：

```text
RCS-PGD vs. I-ADV-RCS
```

这一步只回答：代码是否正确工作，以及 I-ADV 结构在 Radar RCS 上是否比逐点 PGD 显示出继续研究的价值。RCS-FGSM 已经存在，可以保留在结果表中，但不需要为了迁移再次扩展实验。

### 8.2 第二阶段：提升来自哪里（方法有效后必做）

只有第一阶段显示 I-ADV-RCS 有价值后，才运行最小模块消融：

| 方法 | 作用 | 分区 | 梯度增强 | 融合 |
| --- | --- | --- | --- | --- |
| RCS-PGD | 当前直接基线 | 否 | 否 | 逐点梯度 |
| Voxel-Average | 排除“只要分组就有效”的可能 | 是 | 否 | 组内平均 |
| Voxel-Extremum | 单独检验 I-ADV 极值融合 | 是 | 否 | 正负极值比较 |
| I-ADV-RCS | 检验角度—距离梯度增强的额外作用 | 是 | 是 | 正负极值比较 |

### 8.3 原论文完整比较（可选）

只有在需要声称“完整复现 I-ADV 比较协议”时，才补充：

- RCS-Random；
- RCS-MI-FGSM；
- RCS-NI-FGSM；
- Voxel-Random；
- Voxel-Voting；
- 多种检测器和跨模型迁移。

这些不是判断 I-ADV 能否迁移到 4D Radar 的前置条件。

### 8.4 论文实验（方法确定后）

最终论文应选择与研究主张直接相关的强基线，而不是机械复制 I-ADV 的全部表格。届时再根据自己的方法属于 RCS、几何、联合属性还是迁移攻击，确定需要保留哪些对比对象。

### 8.5 公平性要求

公平比较必须固定：

- 相同的 100 个 `uniform` 验证样本；
- 相同 checkpoint 和 PointPillars 配置；
- 相同攻击损失；
- 相同 `epsilon_rcs`；
- 相同的 10 次迭代和 `0.2 * epsilon_rcs` 步长；
- 相同随机种子；
- 相同置信度阈值；
- 相同评价指标；
- 相同攻击区域权限。

第一阶段可以继续使用 100 帧均匀子集，只判断迁移和模块是否有效。只有方法表现出价值后，才需要在完整 1296 帧验证集上运行 VoD 官方 AP。

## 9. 实现验证清单

### 9.1 数据约束

- [ ] 攻击前后每帧点数完全相同。
- [ ] 点顺序和帧归属不变。
- [ ] xyz 完全不变。
- [ ] `v_r` 和 `v_r_comp` 完全不变。
- [ ] time 完全不变。
- [ ] 只有 RCS 发生变化。
- [ ] `max(abs(rcs_adv - rcs_clean))` 不超过预算和数值容差。
- [ ] 对抗点云不存在 NaN 或 Inf。
- [ ] 保存后重新加载的点云与内存结果一致。

### 9.2 算法模块

- [ ] 体素分组能恢复所有输入点，没有重复或遗漏。
- [ ] 同一分组的 RCS 更新方向一致。
- [ ] 单点体素有明确定义且不会报错。
- [ ] `f` 只由 clean xyz 预计算一次，攻击迭代不会意外修改它。
- [ ] `mu` 动量更新发生在反射特征增强之前。
- [ ] 缓存分组与逐轮重新分组产生相同结果。
- [ ] 邻居不足时有可复现的回退规则。
- [ ] PCA 退化时不会产生非有限法向量。
- [ ] 角度和距离特征处于预期范围。
- [ ] 梯度增强关闭时可退化到明确的消融基线。
- [ ] 随机过程受 `seed` 控制。

### 9.3 科学验证

- [ ] I-ADV-RCS 与 RCS-PGD 使用完全相同的损失和预算。
- [ ] 报告分组点数和 PCA 稳定性，而不只报告攻击效果。
- [ ] 分别报告完整方法及 VP、GE、GF 模块消融。
- [ ] 使用 VoD 官方 AP 作为正式主指标。
- [ ] 不把数字攻击结果表述为物理可实现攻击。

## 10. 待确认问题

在编写攻击实现前，必须尽量解决以下问题：

- [x] 原论文攻击分组体素为边长 `0.1 m` 的立方体。
- [ ] KD-tree 的邻居数、半径或搜索规则是什么？
- [ ] 邻居不足和 PCA 退化时如何处理？
- [ ] 法向量方向和 `phi` 的角度约定是什么？
- [x] 原论文 `d_max = 75 m`；Radar 版本仍需决定使用原值还是 VoD 对应范围。
- [x] 原论文梯度增强系数 `lambda = 1000`。
- [x] 原论文运行 10 步，单步 `0.2`，相对 clean intensity 的最大扰动为 `1.0`；投影顺序仍需复核。
- [x] 衰减因子 `mu = 1.0` 用于归一化输入梯度的跨轮动量累积，随后再乘反射特征和 `lambda`。
- [~] 梯度归一化范数未在 I-ADV 公式中标明；正文说明使用 MI-FGSM 框架，因此暂定 L1，并标记为复现假设。
- [ ] 原论文 `L(F(theta, X, y_gt))` 的检测损失具体由哪些项组成？
- [~] 主实验明确只扰动目标车辆，另有 full-scene 实验；目标点通过 GT 框、预先裁剪样本还是其他规则选择仍未说明。
- [x] 原论文 intensity 合法范围为 `[0,1]`；仍需确认每一步的范围裁剪和预算投影顺序。
- [x] ASR 以 Car 实例为样本；KITTI 使用 3D IoU `0.7`，nuScenes 使用 `1 m` 中心距离，说明匹配标准随数据集变化。
- [x] VoD 官方第二组阈值使用 Car 3D IoU `0.5`、Pedestrian/Cyclist 3D IoU `0.25`。
- [x] 截至 2026-08-01 未找到可复用的官方实现，因此也没有可确认的代码许可证；后续若作者发布需重新检查。
- [ ] VoD RCS 是否经过距离补偿或其他归一化？

## 11. 决策记录

实现过程中每项非直接映射都应追加到此表。

| 日期 | 决策 | 原因 | 对公平比较的影响 | 证据 |
| --- | --- | --- | --- | --- |
| 2026-08-01 | 第一阶段仅研究白盒、数字域、非定向、已有点 RCS 修改 | 与当前点云攻击框架一致，并控制研究范围 | 不与点添加、删除或物理攻击直接比较 | 研究计划 |
| 2026-08-01 | 将直接迁移和 Radar-aware 改进拆成两个版本 | 防止把方法适配与创新混在一起 | 可以公平评价 LiDAR 方法迁移效果 | 研究设计 |
| 2026-08-01 | 记录论文第 4.1 节参数，但不直接复制 intensity 数值预算到 RCS | 两种传感器属性的单位和范围不同 | 保留迭代结构与相对步长，Radar 预算单独选择 | 论文第 4.1 节 |
| 2026-08-01 | 新增逐 Car 匹配 ASR，而不使用当前内部 ASR 代替 | 当前内部 ASR 不等价于论文定义 | 具体匹配阈值随后按目标数据集官方协议确定 | 论文第 4.1 节 |
| 2026-08-01 | 按 Algorithm 1 保留“归一化梯度动量 → 反射特征增强 → 体素极值融合”的顺序 | 顺序是 I-ADV 核心，交换会得到不同算法 | RCS-PGD 与 I-ADV-RCS 只在明确模块上不同 | 论文第 3.2 节 Algorithm 1 |
| 2026-08-01 | 区分论文每步更新符号 `epsilon` 与总预算 `eta` | 原论文两个符号语义不同 | 代码用 `step_size` 和 `epsilon_rcs` 分开表达 | 论文公式 (3)、(9) 与第 4.1 节 |
| 2026-08-01 | VoD 主 Car ASR 使用官方 3D IoU `0.5`，而非 KITTI 的 `0.7` | nuScenes 附录已经表明 I-ADV 会按数据集改变匹配标准 | 论文对齐建立在 ASR 逻辑而非照搬阈值上 | 论文附录 A 与 VoD 官方评估代码 |
| 2026-08-01 | 同时实现并明确区分 object-level 与 full-scene 协议 | 原论文主实验只扰动目标车辆，full-scene 是单独实验 | 避免把更强的全场景攻击与论文主表直接比较 | 预印本第 4.3 节 Scene-level perturbation |
| 2026-08-01 | 若无作者代码，梯度归一化暂定 L1 | I-ADV 正文说明采用 MI-FGSM 框架，而标准 MI-FGSM 使用 L1 归一化；I-ADV 自身未标范数 | 配置和结果中必须标注为复现假设，并可做 L1/L2 敏感性实验 | 预印本 Algorithm 1 后正文与 MI-FGSM 原论文 |
| 2026-08-01 | 记录当前未发现官方实现 | 官方页面和公开学术/代码索引均无仓库链接 | 实现属于透明复现，不声称运行作者代码 | OpenReview、ScienceDirect、Crossref、Semantic Scholar、OpenAlex、GitHub 搜索 |

## 12. 完成迁移表的判定标准

满足以下条件后才能开始正式实现：

1. 第 10 节中影响算法复现的关键问题已从论文/代码确认，或已冻结为可配置、可消融且明确标注的复现假设。
2. I-ADV-RCS 的攻击损失、分组方式、更新规则和预算定义没有歧义。
3. 已明确哪些改动属于直接迁移，哪些只能放入 Radar-aware 版本。
4. 已定义最小单元测试和与 RCS-PGD 的公平比较协议。

## 13. 第一版实现记录

已实现：

- 体素化之前的 raw-point RCS-only 攻击，输出仍为对抗点云；
- `gt_boxes` Car 目标区域和 `scene` 全场景两种协议；
- KD-tree 邻域、PCA 法向量、角度和距离反射特征；
- L1/L2 可配置的逐样本梯度归一化、MI-FGSM 动量和梯度增强；
- `0.1 m` 立方体分组与极值梯度融合；
- clean-relative RCS 预算投影和可选 RCS 合法范围；
- VoD Car `Object ASR@3D IoU 0.5`；
- 与 RCS-PGD 同预算、同 10 步和同样本子集的筛选配置；
- 几何、稀疏 PCA 回退、融合方向、RCS-only 和预算约束单元测试。

当前冻结的复现假设：

- `k=16`，查询点计入邻域；
- 少于 3 个有效邻居时令角度因子为 1，退化为 distance-only；
- MI-FGSM 对应的 L1 归一化；
- object-level 点由目标类别 GT 3D 框选取；
- 每轮先做 clean-relative 预算投影，再做可选 RCS 数值范围裁剪；
- 未指定绝对 RCS 上下界时只约束相对扰动。
