# Radar measurement geometry 100-frame screening

日期：2026-08-11

本轮只比较攻击空间和 current-sweep 自由度。所有实验使用相同 uniform 100 帧、
130 个一对一 clean-detected Car、training loss、10 步确定性 PGD、真实 hard
revoxelization，以及完全相同的 `target ∩ time=0 ∩ clean-active` 点集合。当前
measurement epsilon 是 research/digital budget，不是经过验证的物理传感器误差。

## Current-sweep 点数

| 指标 | Mean | Median | P10 | P25 | P75 | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT 内全部点 | 26.108 | 18.5 | 5.9 | 11 | 36.5 | 56 | 0 | 109 |
| time=0 点 | 5.646 | 4 | 1 | 2 | 8 | 12 | 0 | 29 |
| active time=0 点 | 5.638 | 4 | 1 | 2 | 8 | 12 | 0 | 28 |
| current/active-total ratio | 0.235 | 0.200 | 0.124 | 0.167 | 0.270 | 0.369 | 0 | 1 |

active time=0 点为 0、至多 1、至多 2、至多 5 的目标分别占
2.31%、16.15%、29.23%、63.08%。clean hard-voxel capacity 几乎没有继续减少
点数；主要减少来自只允许 time=0。

## 主结果

| Attack | Targets | Mean attacked points | Object ASR | Mean IoU drop | Median IoU drop | Mean score drop | Mean evidence drop | Mean XYZ L2 | Max XYZ L2 | Mean reassignment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XYZ-current ε=0.05 | 130 | 5.638 | 0 | 0.004379 | 0.002075 | 0.003204 | 0.025771 | 0.077849 | 0.086602 | 0.424047 |
| Range-only 0.05 m | 130 | 5.638 | 0 | 0.000623 | 0.000460 | 0.000138 | 0.014280 | 0.043135 | 0.050004 | 0.245758 |
| Azimuth-only 0.1° | 130 | 5.638 | 0 | 0.001283 | 0.000647 | 0.000163 | -0.002381 | 0.035365 | 0.084693 | 0.240371 |
| Elevation-only 0.1° | 130 | 5.638 | 0 | 0.001160 | 0.000475 | 0.000858 | 0.003980 | 0.037883 | 0.084699 | 0.001538 |
| Measurement-all | 130 | 5.638 | 0.007692 | 0.003177 | 0.001493 | 0.003271 | 0.034259 | 0.070550 | 0.129852 | 0.373759 |

唯一成功目标是 localization failure，不是 pure hiding。其 clean IoU 为
0.53566，攻击后同类 IoU 为 0.49978；它有 7 个 active current 点，3 个换柱。
一个成功样本不足以做成功/失败机制推断。

## 位移匹配与机制诊断

辅助 XYZ-current ε=0.025 的 mean/max XYZ L2 为 0.039912/0.043301 m，覆盖了
三个单变量 measurement 的平均位移范围。它的平均 IoU drop 为 0.002149，仍
高于每个 measurement 单变量，但 ASR 同样为 0。XYZ ε=0.05 与 Measurement-all
的平均位移也较接近（0.077849 vs 0.070550 m），两者连续指标有强有弱，不能说
measurement parameterization 已明显摧毁 geometry attack 能力。

换柱率与 IoU drop 的 Spearman rho 在五种主攻击中为 -0.0066 到 0.1080；
Measurement-all 为 0.0487。当前结果不支持“pillar reassignment 是主要攻击机制”。
角度攻击的距离与位移相关性很高：azimuth 0.8316、elevation 0.8828、联合
0.8814，符合固定角度预算在远距离产生更大 Cartesian 位移的几何关系。

## 当前判断

最有直接数据支持的瓶颈是 current-sweep 可攻击点过少：中位数只有 4，且
63.08% 的目标不超过 5 点。下一步优先研究 historical sweep 的时序一致攻击，
而不是立即修改 loss、soft voxelization 或加入 RCS/Doppler。该判断仍需在时序
约束明确后用同一 target-level 指标验证。
