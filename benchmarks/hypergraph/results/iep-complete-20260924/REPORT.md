# IEP 阶段完整 case 验证

日期：2026-09-24。当前可用的 IEP case 已全部闭环：40 个 exact objective 小例、4 个 weighted-capacity 小例、IBM01／IBM02 的 seeds 30–39，以及此前只登记但未运行的 `bad_for_ec`。完整机器可读结果见 [summary.json](summary.json)。

这里的“完整 IEP case”按阶段划分：geometry 60 和 finite-gap 3 属于 OGP／搜索几何诊断，coarsening 400 属于 contraction 与 packing 诊断；它们已有独立结果，但不重复计入 IEP 性能集合。缺少原始输入的 legacy 名称也没有伪装成已完成 case。

## 验证方法

- 对 40 个 objective case，重新从保存的原始超边和 assignment 独立计算 weighted km1、loads 和容量，并核对精确最优值。
- 对 4 个 weighted case，完整枚举所有容量可行的 q-way assignments，重新得到全局最优，再核对 FEM assignment。
- 对 IBM01／IBM02 的 20 个 seed，使用固定本地 `.hgr` 输入重新计算 coarse-lifted 与 final assignment 的 km1 和容量，不直接相信旧 JSON 中的 cut。
- 对 `bad_for_ec`，新运行 direct FEM、HEM／boundary 两种粗化，以及 local-cap／exact-global-guard 两种可行性策略；每个配置使用 seeds 30–39、32 trials × 300 steps。
- 主仓库当前 IEP 相关回归 80 项通过，FEM 子模块 21 项通过，共 101 项。

IBM 和 pathology 输入已从临时目录迁入 `benchmarks/hypergraph/data/hmetis`，并在每次验证时检查 SHA-256。

## 总结果

| case 集合 | case 数 | 执行／审计成功 | 容量可行 | 精确最优命中 |
|---|---:|---:|---:|---:|
| Objective exact | 40 | 40/40 | 40/40 | 40/40 |
| Weighted capacity | 4 | 4/4 | 4/4 | 1/4 |
| IBM01／IBM02 seeds 30–39 | 20 | 20/20 | coarse 与 final 均 20/20 | 无大规模 oracle |
| `bad_for_ec` direct FEM | 10 | 10/10 | 10/10 | 10/10 |

Weighted case 的精确结果保持此前结论：

| case | FEM final km1 | exact optimum | gap |
|---|---:|---:|---:|
| n=8, q=2 | 21 | 17.5 | 3.5 |
| n=8, q=3 | 37 | 37 | 0 |
| n=12, q=2 | 25.5 | 22 | 3.5 |
| n=12, q=3 | 48.5 | 44.5 | 4 |

IBM assignment 的独立审计结果：

| instance | seeds | mean coarse-lifted km1 | mean final km1 | final range |
|---|---:|---:|---:|---:|
| IBM01 | 10 | 1498.2 | 820.1 | 607–927 |
| IBM02 | 10 | 2379.0 | 1317.0 | 838–1880 |

这些 IBM final 值包含与 IEP 起点相同的 classical FM V-cycle，因此用于验证“IEP 初始化经过相同后处理后仍有效”；它们不是 FEM-IER 结果。

## `bad_for_ec` 的新发现

原问题为 n=10、m=15、q=2、epsilon=0.03，精确最优 km1=1。Direct FEM 在 10/10 seeds 上命中最优。

| 粗化路径 | 成功得到可行 coarse IEP | coarse FEM 命中 coarse optimum | lift 后命中原图 optimum | 成功样本 mean lifted km1 |
|---|---:|---:|---:|---:|
| HEM local-cap | 8/10 | 8/8 | 7/8 | 1.75 |
| HEM global-guard | 10/10 | 9/10 | 7/10 | 3.0 |
| boundary local-cap | 8/10 | 6/8 | 6/8 | 3.0 |
| boundary global-guard | 10/10 | 8/10 | 6/10 | 3.6 |

这个 case 将两个问题清楚地区分开：

1. local cluster-cap 不足以保证全局 packing 可行；HEM 和 boundary 都有 2/10 个 seed 产生 coarse-infeasible quotient。
2. exact global guard 将可行率恢复到 10/10，但不能恢复粗化已经丢失的低 cut 划分；因此 guard 解决 feasibility，不自动解决 contraction quality。

`bad_for_ec` 也说明只看“FEM 是否解到 coarse optimum”会高估 IEP：即使 coarse selector 最优，lift 后仍可能因 hierarchy 的表示损失而远离原图最优。

## 完成判据与边界

IEP 阶段的当前可用 case 已满足以下判据：输入固定并有 hash；每个保存 assignment 都重新计算 km1 与 loads；可枚举 case 有全局 oracle；失败以失败类型保留；真实 case 覆盖 10 seeds；缺失原始数据的 legacy case 不计入完成数。

当前证据支持结束 IEP 的实现验证阶段并转向 FEM-IER。它不支持“IEP 在一般超图上已经最优”或“已经超过 KaHyPar”：weighted 4 中只有 1/4 命中全局最优，`bad_for_ec` 还显示 coarse feasibility 与 representational loss 是相互独立的问题。

## 复现

```bash
cd /Users/jonathan/Project/Com_Opt/Hybird-Ising-Partition
/opt/anaconda3/bin/python benchmarks/hypergraph/validate_iep_cases.py \
  --output benchmarks/hypergraph/results/iep-complete-rerun
```

测试命令与输出见 [verification.json](verification.json)。验证脚本拒绝写入非空输出目录，避免覆盖已有结果。
