# 原生 FEM-IER：联合移动选择与 HIP 集成

日期：2026-09-24。实验针对当前修复后的工作树；源文件和输入哈希在运行前后保持一致。原始结果见 [summary.json](summary.json)，每个子目录保存完整候选、选择器、评分、时间和划分。

## 结论

本轮已经把 FEM 接入超图 IER：HIP 负责多层级流程，FEM 在每轮 refinement 中求解联合移动的选择子问题。原生高阶目标、加权可行性和 V-cycle 接口均已接通。

- 从已有 FEM+flow 最终解继续优化，IBM01／IBM02 各 5 seeds 中，FEM-IER 合计改善 5/10，另 5/10 不变，无退化。相对本次随机选择对照为 3 胜、7 平；相对严格时间预算内额外 flow 为 5 胜、5 平。
- 改善幅度很小：IBM01 平均 km1 从 805.6 到 804.8，IBM02 从 1170.2 到 1169.0。20 个 FEM round 中只有 3 个超过共享的 all-off／all-on／single-atom 候选。
- 补充离线诊断后，共享候选加全部两两组合在上述 20 轮中全部追平 FEM，三元枚举也没有额外收益。当前不能把结果解释成 FEM 优于一般组合搜索。
- 在每层 V-cycle 插入 IER 的 seed 30 集成测试中，IBM01 从 806 到 802，IBM02 从 1296 到 1206，但运行更久。这只支持继续研究插入位置，尚未建立等时间优势。
- 小加权例中两个 q=2 起点均达到全局最优，改善来自共享 single-atom 候选，不能归为 FEM 联合选择的优势。n=12、q=3 最后仍比全局最优高 0.5。

没有比较 KaHyPar，没有复现稿件旧实验，也没有建立 SOTA、OGP 突破或近线性可扩展性结论。

## 1. 与原 HIP 的关系

本轮前的验证路径是 `HEM → native FEM 初始化 → flow V-cycle`，属于 IEP 方向。这里的 `flow` 实际为 FM 风格增量贪心，不是 max-flow 求解器。

新增的可选路径为：

```text
HEM → native FEM 初始化
    → 每层投影 → 生成平衡 move atoms → FEM 联合选择 → native 验收 → flow
```

`HyperRefineSolver` 支持 `mode_cycle=('fem_ier',)` 以及与 `flow` 组合；默认仍为 `('flow',)`。FEM 是 HIP 内部求解器，IER 是候选生成、联合选择、验收和候选刷新的循环框架。random／exact 可以替换联合选择后端。

## 2. 原生联合移动目标

每个二元变量选择一个完整 atom。当前候选包括等总重 pair 和 2-for-1 交换；atoms 之间顶点互斥。每个 atom 保持各块负载不变，并检查浮点正残差累积的最坏上界，因此任意子集都满足原容量限制。这比允许负载在容量内变化的完整可行域更严格。

同一 atom 的所有节点共用一个 selector，不能把它们当成独立随机节点。设 selector 开启概率为 r_j，A⁰、A¹ 表示该 atom 在关闭／开启时是否没有节点以标签 b 占据超边 e，Aᶠ 表示固定节点中该标签缺席，则：

```text
P(标签 b 在超边 e 中缺席)
  = Aᶠ[e,b] × ∏j ((1-r_j) A⁰[e,j,b] + r_j A¹[e,j,b])
E[km1] = Σe w_e (Σb (1 - P(缺席)) - 1)
```

此式在不同 selector 独立的变分分布下精确；保留高阶目标交互，但没有学习 selector 之间的相关分布，也不是 Gibbs 采样。目标未做 QUBO 二次截断，不能直接宣称可在仅支持成对作用的 Ising 硬件上运行。

候选池默认从随机 cut net 沿 incidence 扩展，最多 96 个节点、24 个 atoms；有限局部候选仍可能排除改善方向。`interaction_summary` 只记录超边触及多少 atoms，是潜在作用阶数上界，不是非零高阶系数的证明。

每个后端共享 all-off、all-on 和全部单 atom 候选。FEM 使用 8 trials × 100 steps，最后增加 8 个 hard selectors；随机后端增加 800 个 Bernoulli selectors。实际原生评分和容量检查决定接受，只接受严格改善。

## 3. 实验设计与固定起点结果

环境为 CPU、Torch 单线程，BLAS／OMP／MKL 也限制为 1 线程。先单独 warmup。IBM 的 q=4、epsilon=0.03，起点为上一轮保存的 FEM 初始化加 flow V-cycle 最终解，seeds 30–34。A/B 独立复制相同起点，交替执行顺序；运行两轮 IER。首轮候选相同，后续候选可能因当前划分分叉。

随机对照是固定配置对照，**不是等时间或等工作量**。只有额外 flow 使用 FEM-IER 的实测完整时长作为严格预算：从独立初始副本开始，每个两 passes 调用接续自身前次结果；仅接纳截止前完整返回的结果。超时调用保留但排除。本次全部 10 组都有有效 flow 输出，8 组预算内完成 1 次、2 组完成 2 次，没有进一步改善。

| 输入 | seed | 已有 FEM+flow | FEM-IER | random | 预算内额外 flow | FEM-IER 秒 | random 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IBM01 | 30 | 806 | 806 | 806 | 806 | 0.342 | 0.233 |
| IBM01 | 31 | 841 | 841 | 841 | 841 | 0.341 | 0.238 |
| IBM01 | 32 | 672 | 672 | 672 | 672 | 0.354 | 0.239 |
| IBM01 | 33 | 864 | 863 | 864 | 864 | 0.348 | 0.242 |
| IBM01 | 34 | 845 | 842 | 843 | 845 | 0.375 | 0.237 |
| IBM02 | 30 | 1296 | 1296 | 1296 | 1296 | 0.442 | 0.353 |
| IBM02 | 31 | 946 | 945 | 945 | 946 | 0.487 | 0.373 |
| IBM02 | 32 | 1526 | 1525 | 1525 | 1526 | 0.496 | 0.375 |
| IBM02 | 33 | 838 | 838 | 838 | 838 | 0.466 | 0.370 |
| IBM02 | 34 | 1245 | 1241 | 1242 | 1245 | 0.483 | 0.368 |

| 输入 | 起点均值 | FEM-IER 均值 | random 均值 | 额外 flow 均值 | FEM-IER 平均秒 | random 平均秒 |
|---|---:|---:|---:|---:|---:|---:|
| IBM01 | 805.6 | 804.8 | 805.2 | 805.6 | 0.352 | 0.238 |
| IBM02 | 1170.2 | 1169.0 | 1169.2 | 1170.2 | 0.475 | 0.368 |

IBM01 seeds 33、34 与 IBM02 seed 34 中，FEM 选出的组合优于该轮共享确定性候选；其余两次改善由单 atom 候选取得。random 的所有改善都由共享 single-atom 候选提供。winner 来源只是本轮归因，不能证明 FEM 是唯一能找到该组合的算法。

这三轮各比共享候选多降低 1，均选择两个 atoms。其中只有 IBM01 seed 33 出现正协同：起点 864，两个单 atom 分别得到 864、865，联合得到 863。另两轮的改善恰为两个单 atom 改善的加和，不能都称为利用了非线性耦合。

Bernoulli(0.5) 对照偏向稠密选择：24 个 atoms 平均开启 12 个，800 次采样中恰好只开启两个的期望次数仅约 0.013。它不能代表所有随机或任意组合搜索。因此另对已保存池做稀疏组合诊断，独立于上述计时实验。

### 保存池上的稀疏组合诊断

[sparse_probe.json](sparse_probe.json) 从保存的原始起点逐轮重放，枚举共享简单候选、所有 atom pairs 和 triples。24 个变量最多有 276 对、2024 个三元组。评分使用独立原生 pin 计数，赢家另以完整原始划分直接核验；不调用 FEM，也不重新生成候选。所有输入与历史生产文件哈希保持不变。

| 保存轨迹 | 轮数 | 共享候选＋pairs 对原赢家：胜／平／负 | 再加 triples：胜／平／负 |
|---|---:|---:|---:|
| IBM FEM | 20 | 0／20／0 | 0／20／0 |
| IBM random | 20 | 3／17／0 | 3／17／0 |
| weighted FEM | 8 | 0／8／0 | 0／8／0 |
| weighted random | 8 | 0／8／0 | 0／8／0 |

三个 FEM 超过简单候选的组合均可由 pairs 枚举找到；没有发现 FEM 停滞而稀疏枚举可改善的保存轮次。**这说明本批数据尚未展示 FEM 超过稀疏组合搜索的解质量优势。** 它也不证明 FEM 与该搜索的端到端行为相同：稀疏赢家没有作为下一轮起点，等分数时不同标签可能改变后续候选；该诊断未覆盖 V-cycle 内部池，也不是时间匹配的 solver 比较。

## 4. 在 V-cycle 每层插入的结果

seed 30 使用与上一轮相同的 HEM hierarchy 和保存的粗层 FEM 初始标签；完整核对 original-to-coarse 映射与粗节点权重。每个 V-cycle 有 9 次 refinement 调用（包含最后完整图上的额外调用）。

| 输入 | flow 最终 km1 | fem_ier+flow 最终 km1 | 相对降低 | flow 仪表化时长 | fem_ier+flow 仪表化时长 |
|---|---:|---:|---:|---:|---:|
| IBM01 | 806 | 802 | 0.50% | 2.155 s | 6.011 s |
| IBM02 | 1296 | 1206 | 6.94% | 3.491 s | 9.145 s |

时长包含逐层验证和记录，**不是等时间比较**。每一阶段保持可行、原生 cut 不增；这也不意味着不同轨迹的最终质量一般具有单调关系。

IBM01 的 IER 改善全部来自单 atom，未观察到 FEM 超过共享候选。IBM02 在 2873 节点层第二轮，FEM 选择三个 atoms，把 cut 从 1848 降到 1831；共享候选最佳为 1838，联合选择额外贡献为 7。其他 IER 改善均由单 atom 获得。

IBM02 端到端减少的 90 不能全部归因于 FEM 求解器：插入 moves 改变后续 flow 轨迹，当前没有做逐层 deterministic-only／random 消融。该层有超边触及多达 13 个 atoms，但这只是潜在交互统计。

该轮选中的三个单 atom 各自降低 10、5、2，联合降低 17，恰为加和。超过 best single 的 7 因此不是非线性协同的证据。IBM01／IBM02 沿新轨迹的 IER 直接累计降低分别为 10／40；这些总量不能直接作为最终两条不同轨迹差值的因果分解。

## 5. 小加权实例与精确基线

使用上一轮四个保存的 FEM-final 起点，保持原始节点／超边权重、严格容量 epsilon=0。原图完整可行枚举给出全局最优。

| 实例 | 起点 | 全局最优 | FEM-IER | random | 最终最优差距 |
|---|---:|---:|---:|---:|---:|
| n=8, q=2 | 21 | 17.5 | 17.5 | 17.5 | 0 |
| n=8, q=3 | 37 | 37 | 37 | 37 | 0 |
| n=12, q=2 | 25.5 | 22 | 22 | 22 | 0 |
| n=12, q=3 | 48.5 | 44.5 | 45 | 45 | 0.5 |

三次改善均来自 single-atom 候选，两个后端的结果完全相同。说明等总重联合移动能扩展原来的局部可达集合；本批小例没有呈现 FEM 联合选择的额外收益。每轮最多 5 个 selectors，随机 800 次可能覆盖全部有限子空间，不能把这种随机对照推广到 24-variable 子问题。

独立审计进一步穷举四个 weighted cases 两个后端的全部 16 个 round 子空间（共 280 个 selector 状态），所有返回值都是对应候选池的精确最优。n=12、q=3 的两轮池最优均为 45，原图最优为 44.5；这 0.5 在本次运行中是候选池缺口，不是 FEM 没解好这些子问题。

另有固定协同 fixture 的回归测试：起点 cost=4，前两个单 atom 各自使 cost=8，第三个使 cost=24，全部开启为 20；FEM 选择前两个、关闭第三个可得 0。它验证 solver 确实能利用组合效应，不是一般性能证据。`exact` 后端和独立枚举对照仅认证生成的 selector 子空间，不能认证原图最优。

## 6. 验证、复现与限制

本轮相关回归测试 **138 passed in 1.31s**，覆盖实际 FEM 协同选择、原生联合期望／梯度、容量可行性、候选生成、加权 refinement、V-cycle、native FEM、quotient 和截止时间语义。

[独立审计](audit.json) 通过，未调用生产目标或 benchmark evaluator：以实际应用 moves 后的原始标签和超边计数核验 **56 rounds、23,684 个记录的候选分数**，其中去重后实际重算 16,387 个 round-selector 状态。另核验 120 个保存的原始划分记录、2 个粗层输入、29 个源文件及 31 个输入哈希、14 组首轮 A/B 候选一致性、22 个 flow 调用及其 12 个截止前有效输出。原图四个小实例重新遍历 542,354 个标签状态，得到 14,418 个可行状态并评分。

审计限制：V-cycle 的中间层超图和每次 flow 后完整划分没有保存，36 个内部 IER rounds 无法从产物独立完全重放。审计核验了原始图上的起点／最终输出，内部阶段记录仅检查单调性及容量一致性；运行时另有逐层检查。IBM 的最多 24 个 selector 未做全空间最优认证。

审计脚本为 `benchmarks/hypergraph/audit_fem_ier.py`，其版本、运行环境和自身哈希记录在 audit.json。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -m pytest -q \
  tests/test_hyper_fem_ier.py tests/test_hyper_ier_objective.py \
  tests/test_hyper_ier_candidates.py tests/test_hyper_weighted_refine.py \
  tests/test_hyper_fem_native.py tests/test_hyper_quotient.py \
  tests/test_fem_ier_benchmark.py tests/test_fem_multiseed_benchmark.py \
  tests/test_time_budget_hgr.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python \
  benchmarks/hypergraph/validate_fem_ier.py --seeds 30 31 32 33 34 \
  --rounds 2 --output benchmarks/hypergraph/results/fem-ier-new-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/anaconda3/bin/python benchmarks/hypergraph/audit_fem_ier.py \
  benchmarks/hypergraph/results/fem-ier-20260924

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/anaconda3/bin/python benchmarks/hypergraph/probe_ier_sparse_selectors.py \
  --results benchmarks/hypergraph/results/fem-ier-20260924 \
  --output /private/tmp/fem-ier-sparse-new-run.json
```

运行要求此前保存的 `fem-multiseed-20260924-v2`、`fem-repair-20260924` 和原 IBM `.hgr` 数据。可用 `--baseline`、`--repair`、`--ibm-directory` 指定位置；程序校验数据哈希，拒绝覆盖非空输出目录。当前原始数据位于 `/private/tmp/ising-hgr.K5jm2Y`，临时目录不保证长期存在。

优先的后续实验是：固定相同 hierarchy／粗层起点，增加 seeds，对比每层单 atom 顺序贪心、稀疏组合搜索、FEM，并匹配完整 wall-time；同时认证小候选子空间最优，区分候选不足和选择器求解损失。随后再比较 KaHyPar 和更广数据集。

这些结果揭示具体邻域与联合选择的作用，没有测量 IBM 近优解 overlap，也没有认证其 OGP。当前 mean-field FEM 仍不能称为已验证的 SOTA 实现。
