# 超图 FEM / PUBO 修复与验证

日期：2026-09-24。修复对象是当前 `FemCoarsenSolver → HyperRefineSolver → vcycle_uncoarsen` 生产 API。此前的 [OGP 诊断](../ogp-diagnostic-20260924/REPORT.md) 保留不变。

已修复入口失效、FEM 缺少目标梯度更新、超图目标错配、加权容量检查和 V-cycle 权重传递等问题。相关测试共 **93 项通过**；当前生产实现通过 40 个保存小实例、4 个加权流程和 2 个 IBM 超图验证。这些结果确认了实现修复，尚不构成 OGP 存在或被突破的证据。

## 修复内容

| 问题 | 修复后的行为 |
|---|---|
| 超图入口引用已删除的 `src.fem` | 从实际 `fem` 子模块包导入；FEM 和 PUBO 入口均可运行 |
| `manual_grad=False` 没有根据目标更新参数 | 对 categorical logits 执行真实 autograd 优化，支持 Adam / SGD / RMSprop；手动梯度明确约定为 `dE/dp` |
| 默认 clique 目标与原生超图目标不一致 | 默认优化 weighted km1 的精确独立分布期望；`method='pubo'` 为原生目标别名，支持任意 `q >= 2` |
| 多候选返回与离散代价不一致 | 每个 trial 经容量检查后，以原生 weighted km1 选择最佳可行候选；保存候选代价和失败记录 |
| soft balance 或简单 rounding 可留下不可行解 | 按物理节点权重检查每块容量；使用多种 greedy packing 和有预算的精确回退；区分已证明不可行与搜索未找到 |
| 资源单位、star 辅助节点干扰平衡 | 软惩罚按逻辑节点平均权重归一化；star 辅助节点资源权重为零；容量容差随总资源缩放 |
| refinement / V-cycle 丢失权重或误施加下界 | 逐层传递节点和超边权重，仅检查定义中的容量上限；必要时执行有检查的容量修复 |

原生目标为

\[
E(p)=\sum_e w_e\left[\sum_{b=1}^q\left(1-\prod_{i\in e}(1-p_{ib})\right)-1\right],
\]

对应离散目标 `sum_e w_e * (number_of_blocks_touched(e) - 1)`。空超边贡献为零，重复 pins 去重。它是独立 categorical 分布下的精确期望，并不代表对所有解分布的精确建模。q=2 时 km1 等于 cut-net；q>2 时两者不同。

FEM 的显式 `clique` / `star` 模式仍可用于代理目标对照，但输出统一按原生 km1 评分。底层还修复了 categorical entropy、inverse 退火端点、QUBO 对角项期望与离散结果返回契约。

## 回归验证

[verification.json](verification.json) 保存命令和输出。

| 测试范围 | 结果 |
|---|---:|
| 主仓库：原生目标/梯度、容量 rounding、加权 refinement、quotient、hGR 与 OGP 诊断回归 | 72 passed |
| `lib/qubo-solver`：FEM 梯度与退火 | 21 passed |

关键断言包括：与完整 categorical 枚举的目标和梯度一致；资源单位缩放不改变容量判定；q>2 的合法非对称 loads 不被误拒；加权 refinement 不因漏掉超边权重而增加真实代价；搜索预算耗尽不误报不可行。主仓库及子模块 `git diff --check` 均通过。

## 当前生产实现的实测

运行 [validate_fem_repair.py](../../validate_fem_repair.py)，直接调用修复后的 API。环境为 macOS arm64、Python 3.12.7、NumPy 1.26.4、PyTorch 2.9.0、CPU 单 Torch 线程。记录的源文件在运行前后哈希一致，完成后再次核对仍一致。

### 保存的小实例

复用此前诊断保存的 40 个实例：n=12 / 16、random / planted 四组，每组 10 个。q=2、严格等分，32 trials × 300 steps；枚举重新计算最优值。

- **40/40 个 best-of-32 返回解可行并达到枚举最优，native gap 全为 0。**这不表示全部 trial 最优；逐候选计为 1254/1280 最优。
- 四组各取一个实例检查 FEM / PUBO：同 seed、同参数下，4/4 完全同解。
- 完整结果见 [saved_instances.json](saved_instances.json)。这些实例规模较小且来自已有诊断集，不能外推成功率或渐近复杂度。

### 加权流程

节点权重交替为 1 / 2，超边权重为正的非单位权重；q=2 / 3、epsilon=0。小规模粗化启用精确全局可行性检查。FEM 使用 32 trials × 300 steps，随后使用 flow refinement。四条 FEM 流程均从可行粗解开始、以可行细解结束；粗解提升到原图的 weighted km1 和各块 loads 守恒，逐层容量检查通过。

| 实例 | Greedy 粗解 km1 | Greedy 粗解可行 | Greedy 最终 km1 | FEM 粗解 / 最终 km1 |
|---|---:|:---:|---:|---:|
| n=8, q=2 | 16.5 | 否 | 17.5 | 21 / 21 |
| n=8, q=3 | 37 | 是 | 37 | 37 / 37 |
| n=12, q=2 | 26 | 否 | 29 | 25.5 / 25.5 |
| n=12, q=3 | 46.5 | 是 | 46.5 | 48.5 / 48.5 |

两条 greedy 粗解分别有 loads `[7,5]` 和 `[10,8]`，容量修复后为 `[6,6]` 和 `[9,9]`；修复可增加 cut，因此只对可行的 refinement 起点断言目标不增加。加权小例明确显示 FEM 并不总胜过 greedy，本组没有证明原始加权问题的全局最优。

### IBM 真实超图

IBM01 有 12,752 个节点 / 14,111 条超边，IBM02 有 19,601 个节点 / 19,584 条超边。原图均为无权输入，粗化后保留聚合权重。

两种初始化共用同一 HEM hierarchy：seed=30、q=4、coarsen_to=200、epsilon=0.03。FEM 为 8 trials × 150 steps；每次 flow refinement 使用 2 passes，8 层解粗化加最后原图处理共调用 9 次。epsilon 仅限制各块负载不超过平均值的 1.03 倍，不施加 0.97 倍的下限。真实数据未启用小规模精确全局粗化检查。

| 实例 | 初始化方法 | 粗解 km1 | 最终 km1 | 初始化耗时 | V-cycle 耗时 |
|---|---|---:|---:|---:|---:|
| IBM01 | Greedy | 3037 | 1180 | 0.043 s | 1.556 s |
| IBM01 | FEM | 1571 | 806 | 0.679 s | 2.017 s |
| IBM02 | Greedy | 9519 | 1920 | 0.087 s | 5.015 s |
| IBM02 | FEM | 2431 | 1296 | 1.466 s | 3.243 s |

共享粗化耗时分别为 0.254 s 和 0.570 s，未计入上表；V-cycle 计时包含诊断记录和验证开销。FEM 最终 km1 相对该 greedy 基线分别降低约 31.7% / 32.5%。两次 FEM 的全部 8 个候选投影均成功；两种方法各层 refinement 后均满足容量约束。

FEM 最终 loads：IBM01 为 `[3234,3201,3280,3037]`，每块容量 3283.64；IBM02 为 `[4720,4787,5047,5047]`，容量 5047.2575。详细逐层数据、时间、候选与 assignments 见 [summary.json](summary.json)、[ibm01.json](ibm01.json)、[ibm02.json](ibm02.json) 及同目录 NPZ 文件。

这是单 seed、单配置的生产流程验证。表中比较的是修复后两种初始化方法，不是失效旧 FEM 与新 FEM 的性能比较，也不是等时间预算或 KaHyPar 对比。

## 复现

在主仓库目录运行主仓库测试；子模块测试在独立进程和独立工作目录运行，避免两个 `src` 包互相遮蔽：

```bash
cd /Users/jonathan/Project/Com_Opt/Hybird-Ising-Partition
/opt/anaconda3/bin/python -m pytest -q tests/test_hyper_fem_native.py tests/test_hyper_weighted_refine.py tests/test_hyper_quotient.py tests/test_time_budget_hgr.py tests/test_ogp_exact_geometry.py tests/test_ogp_objective_diagnostic.py
```

```bash
cd /Users/jonathan/Project/Com_Opt/Hybird-Ising-Partition/lib/qubo-solver
/opt/anaconda3/bin/python -m pytest -q tests/test_fem_gradients.py tests/test_adaptive_annealing.py
```

生产流程验证（输出目录须为新目录，脚本拒绝覆盖已有 `summary.json`）：

```bash
cd /Users/jonathan/Project/Com_Opt/Hybird-Ising-Partition
/opt/anaconda3/bin/python benchmarks/hypergraph/validate_fem_repair.py \
  --saved-instances benchmarks/hypergraph/results/ogp-diagnostic-20260924/objective/instances.jsonl \
  --ibm-directory /private/tmp/ising-hgr.K5jm2Y \
  --output benchmarks/hypergraph/results/fem-repair-rerun
```

IBM 输入位于临时目录，不随仓库分发；该目录清理后需重新准备 `ibm01.hgr` / `ibm02.hgr` 并修改 `--ibm-directory`。固定版本下载链接见 [time_budget_hgr.py](../../time_budget_hgr.py) 文件头，实际输入 SHA-256 见本次 `summary.json`。保存的 40 个小实例文件也是复现依赖。耗时随硬件和运行环境变化。

## 保留的边界

- 一般加权 packing 是困难问题。默认精确回退仅覆盖不超过 20 个节点、最多 200,000 次搜索访问；大实例或预算耗尽可返回 `BalanceSearchError`，不会伪称 UNSAT。粗化已排除的可行解也无法由 rounding 恢复。
- 非单位超边权重使用 `mode_cycle=('flow',)`；hybrid / MCTS / evolution 目前显式拒绝此输入。独立分布目标的中间张量规模随 batch × pins × q 增长，尚未分块处理。
- 底层 FEM 当前实现 `qubo` / `customize`；未实现的旧命名问题明确报错。`use_compile=True` 会警告并使用 eager。这次未迁移全部旧图 benchmark 或修复其外部数据依赖。
- 修复包含 `lib/qubo-solver` Git 子模块中的本地修改。尚未提交或推送；复现时需要保留主仓库和子模块两处修改，仅切回原子模块提交不会包含这些修复。
- 下一阶段若研究 OGP，应在已验证的原生目标、可行性和权重契约上开展多 seed、等预算、不同搜索方法的对照。本次不能将既往实现错误或有限小实例的 overlap 缺口解释为 OGP 算法障碍。
