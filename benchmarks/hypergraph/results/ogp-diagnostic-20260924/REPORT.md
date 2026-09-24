# HIP 超图诊断实验：先排除实现与表示问题

日期：2026-09-24。

本轮完成三组小规模可复现实验：目标与梯度、固定粗化层级的精确对照、原生近优集合的完整 overlap 枚举。结果支持优先处理实现和建模问题，尚不支持用 OGP 解释稿件中的超图性能退化。

**范围**：当前工作区上的诊断，保留已有未提交改动；未修改生产求解器或 submodule。合成实例用于隔离机制，不是论文真实数据集、KaHyPar 对比或 HIP 端到端复现。各模块使用不同的显式实例配置，不能把它们当作同一批数据合并推断。

## 1. 实际运行发现：现有 FEM 管线尚不能作为有效对照

- `FemCoarsenSolver.initial_partition` 的 `fem`、`pubo` 两个分支均实际触发 `ModuleNotFoundError: No module named 'src.fem'`。求解器已迁移到 `lib/qubo-solver/src/fem`，旧入口没有同步。
- 独立调用外部 `FEM` 的 `manual_grad=False` 分支，在相同初始概率下，分别用 `+sum(p1)` 和 `-7*sum(p1)` 两个相反方向的目标运行 12 步：最终概率逐位相同，最大差为 **0**。两者也都与仅重复 softmax 12 次完全相同，最大误差为 **0**。当前源代码该分支没有执行 autograd 更新。
- 因此不能把这次旧入口失败、或该分支的输出质量，解释成 OGP 障碍。也不能用这些当前版本的问题反推稿件历史实验一定运行了相同代码。

运行记录：`objective/summary.json` 的 `runtime_audit`；源文件哈希随结果保存。

## 2. 原生目标与梯度核验

验证的是节点独立类别分布下的原生 km1 期望：

$$\mathbb E[C]=\sum_e w_e\left[\sum_{b=1}^k\left(1-\prod_{i\in e}(1-p_{ib})\right)-1\right].$$

对四节点、k=2/3 的加权超图，枚举全部类别赋值后，期望误差最大为 `8.9e-16`，对 logits 的梯度误差最大为 `2.2e-16`；PyTorch 双精度有限差分 `gradcheck` 通过。包含 singleton 超边和多分区 km1≠cut-net 的检查。
原 PUBO cut 使用的 `argmax → NumPy → scalar` 表达式单独核验后不含 autograd 连接；这是表达式层面的检查，并非成功运行了旧 PUBO 管线。

## 3. 即使精确求解，clique 也可能选错原生最优解

单位节点/超边权重、严格平衡二分；n=12/16，各 random/planted 10 个实例，每实例 2n 条四节点超边（重复超边保留），planted 内部抽样概率 0.85。随机置换顶点编号以隐藏 planted 的连续标签次序。完整枚举全部可行划分并消除整体翻转。
clique 使用仓库 `build_clique_expanded_graph` 一致的每对权重 `1/(|e|-1)=1/3`；这是明确指定的 normalized clique 对照，不是已失效的历史 helper 的复现。

| 实例族 | n | 实例数 | 所有 clique 最优解均为原生次优 | 至少一个 clique 最优解为原生次优 | clique 最优集合中最佳原生 gap 的均值 |
|---|---:|---:|---:|---:|---:|
| random | 12 | 10 | 1 | 5 | 0.200 |
| planted | 12 | 10 | 0 | 0 | 0.000 |
| random | 16 | 10 | 3 | 5 | 0.300 |
| planted | 16 | 10 | 0 | 0 | 0.000 |

**关键结果：20 个 random 实例中，4 个实例的所有 clique 最优解在原生目标上都严格次优。** 这个差异由精确枚举确认，不能归咎于启发式没有把 clique 解好。

## 4. 独立 autograd mean-field 参考实验

为绕开上述运行阻断，新增了明确标注的独立 PyTorch 参考优化器，未将它冒称为当前 HIP/FEM 实现。三个目标分别是原生期望、normalized clique 期望、切断 cut 梯度的控制组；同一实例逐位相同的 32 组随机 logits 初始化，300 步 Adam，学习率 0.08，温度由 2.0 指数降至 0.02，期望负载偏差平方罚系数为 5。CPU 单线程、float64。
最终先记录原始 0.5 threshold 的可行率，再用同样的 top-k 舍入强制严格二分平衡，并记录原生 cut。最后对每个舍入候选执行相同的、仅接受严格改进的最佳 1-for-1 swap descent。
下表 gap 均指原生 cut 减去精确最优；每实例先取 32 候选中的 best，再对 10 个实例求均值。

| 实例族 | n | 参考目标 | 原始 threshold 可行率 | top-k 后平均 best gap | top-k 命中最优实例 | swap 后平均 best gap | swap 后命中最优实例 |
|---|---:|---|---:|---:|---:|---:|---:|
| random | 12 | native | 88.4% | 0.000 | 10/10 | 0.000 | 10/10 |
| random | 12 | clique | 43.1% | 0.300 | 8/10 | 0.000 | 10/10 |
| random | 12 | detached_cut | 30.6% | 1.300 | 1/10 | 0.000 | 10/10 |
| planted | 12 | native | 100.0% | 0.000 | 10/10 | 0.000 | 10/10 |
| planted | 12 | clique | 100.0% | 0.000 | 10/10 | 0.000 | 10/10 |
| planted | 12 | detached_cut | 32.2% | 10.600 | 2/10 | 0.000 | 10/10 |
| random | 16 | native | 52.8% | 0.100 | 9/10 | 0.000 | 10/10 |
| random | 16 | clique | 62.8% | 0.300 | 7/10 | 0.000 | 10/10 |
| random | 16 | detached_cut | 31.6% | 3.100 | 1/10 | 0.000 | 10/10 |
| planted | 16 | native | 100.0% | 0.000 | 10/10 | 0.000 | 10/10 |
| planted | 16 | clique | 100.0% | 0.000 | 10/10 | 0.000 | 10/10 |
| planted | 16 | detached_cut | 31.2% | 17.500 | 0/10 | 0.000 | 10/10 |

- random 实例中，原生目标在 top-k 后命中精确最优 **19/20**，clique 为 **15/20**，切断 cut 梯度为 **2/20**。
- 随机初始概率直接 top-k 的平均 best gap：random n=12 为 1.3，n=16 为 3.0；原生目标训练后对应 0 和 0.1。
- 相同 swap refinement 后，三组在全部 40 个实例上都至少有一个候选达到最优。初始化质量差异仍可从候选平均 gap 看出，但只看最终 best cut 会遮蔽这些差异。
- 原始 threshold 可行率说明：期望负载罚项并不保证离散可行性。这里的精确平衡由 top-k 保证；它不直接适用于一般加权粗节点。
- 固定参数是配对控制，没有分别为各目标调参；目标尺度不同，且比较固定步数而非等 wall time。不能推导一般性能优越性。`detached_cut` 是控制组，不是原 PUBO 的完整复现。
- `solution_pool.jsonl` 保存了全部 3,840 个末态概率以及 raw/rounded/refined 划分和原生代价；这是三个目标×40实例×32候选的记录，含重复解，不代表3,840个独立实例。

## 5. 固定 hierarchy 的粗化损失与可行性

复用仓库已有粗化 benchmark，每个实例族 200 个实例、seed=4000..4199，n=8、14 条大小2–4的超边，目标粗化到4节点。原始问题和每个粗层均完整枚举求解，排除 FEM/SBM 的求解误差。

单位节点权重、容量容差 ε=0.25：

| 实例族 | 原始最优均值 | HEM 粗层精确最优均值 | boundary 粗层精确最优均值 | HEM 粗化损失 | boundary 粗化损失 |
|---|---:|---:|---:|---:|---:|
| random | 6.975 | 8.310 | 8.325 | 1.335 | 1.350 |
| planted | 1.600 | 1.825 | 1.805 | 0.225 | 0.205 |

boundary 相对 HEM 的平均改善（HEM cut − boundary cut）：random −0.015，配对正态近似95%区间 [−0.0865, 0.0565]；planted 0.020，区间 [−0.0645, 0.1045]。两个区间均跨零，此样本不支持稳定改进。

随机节点权重1–4、原始实例经拒绝采样确保严格平衡可行，ε=0：

| 实例族 | 方法 | 粗层不可行数 | 比例 |
|---|---|---:|---:|
| random | hem_local | 76/200 | 38.0% |
| random | hem_guarded | 0/200 | 0.0% |
| random | boundary_local | 70/200 | 35.0% |
| random | boundary_guarded | 0/200 | 0.0% |
| planted | hem_local | 97/200 | 48.5% |
| planted | hem_guarded | 0/200 | 0.0% |
| planted | boundary_local | 94/200 | 47.0% |
| planted | boundary_guarded | 0/200 | 0.0% |

exact global guard 使用小规模指数时间装箱检查；它提供可行性 oracle，不是大规模可扩展性的证明。仅比较 local 可行样本上的 conditional mean 与 guarded 全样本均值会产生选择偏差。

## 6. 完整 overlap 几何检验

原始超图、严格平衡、n=8/12/16、random/planted 各10实例，共60实例；每个实例2n条四节点超边，planted概率0.75，seed base=20260924；绝对近优阈值 ε=0/1/2。与上面的目标对照使用不同 seeds 和 planted 强度。
完整可行划分在去标签翻转后分别为35、462、6,435个；不同解类的全部无序对分别为595、106,491、20,701,395。近优集合和近优解对均精确枚举，没有用启发式样本替代全集。
180 个“实例×阈值”组合中，两端均由不同解类配对支持的保守缺口为 **0/180**；允许标准 all-pairs 定义中的 q=1 自配对、同时排除单解集合后，有限实例缺口为 **3/180**。前一种保守条件不是 OGP 的必要条件，不能把 0/180 简写成“没有 OGP”。

| 实例 | 阈值 | 最优解类数量 | 含自配对的 overlap 支持 | 缺失的可行格点 |
|---|---:|---:|---|---|
| random-n08-i003 | ε=0 | 2 | {0, 1} | 1/2 |
| random-n08-i009 | ε=0 | 2 | {0, 1} | 1/2 |
| random-n12-i008 | ε=0 | 2 | {0, 1} | 1/3、2/3 |

其余 planted 的 90 个实例阈值组合均只有一个近优解类，不作为非平凡分簇证据。累计精确计数 237,596 个近优不同类解对；阈值集合嵌套，因此这不是互不重复的解对数量。

见 [geometry/summary.md](geometry/summary.md) 的完整统计。这三个缺口都来自极窄能量窗口中的两个最优划分；有限 gap 和未观测到 gap 都不能据此建立或排除渐近 OGP，更不能直接推出算法下界。

### 在这三个有限缺口实例上直接求解

使用同一个独立参考优化器，对上述3个已完整枚举的实例运行32组同 seed 初始化、300步，并做严格改善的原生1-for-1 swap refinement。先重新枚举并逐项核对了保存的完整 mask–energy 字典。

| 实例 | 原生最优 | 参考目标 | top-k 最优命中 | top-k best gap | swap 后最优命中 | swap 后最优类覆盖 |
|---|---:|---|---:|---:|---:|---:|
| random-n08-i003 | 14 | native | 5/32 | 0 | 32/32 | 2/2 |
| random-n08-i003 | 14 | clique | 0/32 | 1 | 32/32 | 2/2 |
| random-n08-i003 | 14 | detached_cut | 3/32 | 0 | 26/32 | 2/2 |
| random-n08-i009 | 14 | native | 0/32 | 1 | 32/32 | 1/2 |
| random-n08-i009 | 14 | clique | 0/32 | 1 | 32/32 | 1/2 |
| random-n08-i009 | 14 | detached_cut | 1/32 | 0 | 24/32 | 2/2 |
| random-n12-i008 | 19 | native | 1/32 | 0 | 32/32 | 2/2 |
| random-n12-i008 | 19 | clique | 0/32 | 1 | 32/32 | 1/2 |
| random-n12-i008 | 19 | detached_cut | 0/32 | 1 | 23/32 | 2/2 |

**所有9组“实例×方法”在 swap 后的 best gap 都为零。** 这些有限 overlap 缺口没有阻止本轮简单 refinement 找到最优。优化路径可以经过非最优状态；它不需要沿着仅由最优解组成的空间连续移动。
最优解类覆盖并不总是完整，且没有估计 Gibbs 权重或 mixing time。因此，找到最优、覆盖多个最优类、忠实采样分布必须分开评价。这仅是3个小实例的实测，不是任意算法或渐近 OGP 的结论。完整逐trial结果与源文件哈希见 `crosscheck/summary.json`。

## 7. 对后续工作的含义

1. 先修复旧 FEM 导入/API 与实际 autograd 更新路径，建立有效的原生目标求解入口。
2. 粗化必须分开评价保留最优解的能力和全局负载可行性；本轮已经测出两者的损失。
3. 原生期望适合作为基线，但需要保留离散可行性处理和 refinement 对照；当前结果不支持宣称单靠它就解决了超图划分。
4. 接下来在真实 hgr 上以相同 hierarchy、候选与时间预算比较，才能回答稿件性能退化是否被修复。
5. 只有明确原生目标、实例族、阈值、overlap 定义和适用算法类后，才讨论 OGP 障碍。

## 复现与产物

本机解释器：`/opt/anaconda3/bin/python`；Python 3.12.7、NumPy 1.26.4、PyTorch 2.9.0。仓库 HEAD 为 `6008ff62adea4f24492ce0e2d9610a95036d92a0`，工作区已有未提交修改；结果中记录相应源文件 SHA-256。

相关验证共 **32 项测试通过**（原有 quotient 14 项、exact geometry 13 项、objective diagnostic 5 项），见 `verification.json`／`verification.log`。额外核对了目标实验的源文件哈希，以及全部 3,840 个候选在舍入和 refinement 后的严格平衡与 refinement 不增代价。

从仓库根目录运行：

```sh
/opt/anaconda3/bin/python benchmarks/hypergraph/ogp_objective_diagnostic.py --instances 10 --sizes 12 16 --trials 32 --steps 300 --seed 4000
/opt/anaconda3/bin/python benchmarks/hypergraph/ogp_coarsen_diagnostic.py --instances 200 --seed 4000
/opt/anaconda3/bin/python benchmarks/hypergraph/ogp_exact_geometry.py
/opt/anaconda3/bin/python benchmarks/hypergraph/ogp_gap_solver_probe.py --output /private/tmp/ogp-gap-recheck
```

- `objective/`：公式/梯度与运行路径记录、逐实例精确目标比较、逐方法 CSV、完整候选池和实例。
- `coarsen/`：已有粗化基线复现、测试日志、配置与源文件哈希。
- `geometry/`：所有可行划分及能量、near-optimal pair histograms、可行 overlap 格点基线和验证记录。
- `crosscheck/`：在3个有限缺口实例上直接运行参考求解器及 refinement 的逐trial结果。
- 四个脚本都位于 `benchmarks/hypergraph/`；没有修改生产求解器。最后一条复现命令需选择尚不存在结果文件的新输出目录。
