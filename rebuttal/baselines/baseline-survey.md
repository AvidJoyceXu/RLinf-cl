# Parameter-space merging baselines：RETAIN 与 TIES 的调研与可用性评估

对应 AC.K4（「缺少标准 model-merging baseline」）。日期 2026-08-12。
材料来源：`rebuttal/baselines/RETAIN/2512.08333v3.pdf` + `RETAIN_code`（openpi 分支），
`rebuttal/baselines/TIES/ties-merging`（官方实现）。方法描述以**代码**为准，论文用于确认意图。

---

## 1. RETAIN

### 1.1 方法

**核心就一个式子**（论文 Eq. 2）：

```
θ̃ = (1 − α) · θ_pre + α · θ_ft
```

在预训练 generalist 权重和微调后权重之间做**线性插值**，α 是可调的融合系数。代码里就是
`src/openpi/policies/model_merging.py::linear_interpolation`，逐参数加权求和，无训练、无额外推理开销。

三个增量：

| 变体 | 内容 | 我们是否需要 |
|---|---|---|
| RETAIN-task-FT | 只在目标任务数据上微调后插值 | 主对比 |
| RETAIN-co-FT | 微调时混入预训练数据 `D_pre` | 需要 LIBERO 预训练集，**可选** |
| Modality-specific | 只融合语言模型 backbone 的参数，视觉/action head 不动 | 消融，**可选** |

**持续学习变体**（论文 Eq. 4）——这一条才是和我们直接可比的：

```
θ̃_n = (1 − α) · θ̃_{n−1} + α · θ_ft,n        n = 1…N
```

即把每个新任务微调出的权重**顺序地**融进不断累积的 checkpoint。注意它是「新任务 vs 当前累积模型」
的两两插值，**没有任何门控**——不判断该不该合并，一律合并。这正好是我们 RFC-gated 的对照组。

### 1.2 在我们 case 上的可用性

RETAIN 融合的是**VLA 主干权重** `θ_pre`（generalist）和 `θ_ft`（在目标任务上微调）。我们的方法
**根本不动 VLA**，只训练一个外挂 residual policy。所以存在一个结构性错配，有两种对齐方式：

**(A) 忠实复现 RETAIN（贵）**
需要为 5 个 object 任务各做一次 VLA 全参/LoRA 微调，得到 `θ_ft,1…θ_ft,9`，再按 Eq. 4 顺序插值。
- 需要：**每个任务的 LIBERO demo 数据**，以及 5 次 7B VLA 微调（或你直接提供的 SFT checkpoint）
- `θ_pre` 用哪个是个需要定的问题：我们现在的 base 是 `Openvla-oft-SFT-libero-object-traj1`，
  它**本身已经在 object suite 上 SFT 过**，不是干净的 generalist。用它当 `θ_pre` 会让 RETAIN
  的「保留通用能力」这一卖点无从体现（因为它的「通用能力」就是这 5 个任务本身）。
  更合适的是用未在 object 上微调过的 OpenVLA-OFT 通用 checkpoint。**这一点需要你确认。**

**(B) 同层对比（便宜，建议先做）**
把 RETAIN 的插值算子搬到 **residual policy 层面**：`θ̃_n = (1−α)·θ̃_{n−1} + α·θ_res,n`。
- 需要：**只需现有的 5 个 residual checkpoint**，无需任何训练
- 回答的问题是「在同一套 residual 上，我们的 merge 算子 vs 朴素线性插值」，
  正好对上 AC.K4 想问的「你的 merge 有什么特别」
- 局限：这不是 RETAIN 原文的设定，写进 rebuttal 时必须说明是「RETAIN 的融合算子迁移到 residual 空间」

### 1.3 成本

| | 训练 | 评测 |
|---|---|---|
| (A) 忠实复现 | 5 × VLA 微调（若你提供 SFT checkpoint 则为 0） | 5×5 SR 矩阵 × α 网格 |
| (B) 同层对比 | 0 | 5×5 SR 矩阵 × α 网格 |

α 需要扫描（论文用 0.1–0.9），每个 α 一张 5×5 表。

---

## 2. TIES-Merging

### 2.1 方法

对每个任务算 **task vector** `τ_i = θ_i − θ_pre`，然后三步（代码 `src/utils/merge_utils.py`）：

1. **Trim**（`topk_values_mask`）：每个 task vector 按**幅度**只保留 top-K%（默认 K=20%），其余置零
2. **Elect sign**（`resolve_sign`）：对每个参数位置，取各 task vector 之和的符号
   （`mass` 模式：`sign(Σ_i τ_i)`）作为该位置的「当选符号」
3. **Disjoint merge**（`disjoint_merge`）：只聚合与当选符号一致的项，对非零项取均值

最终 `θ_merged = θ_pre + λ · τ_merged`，λ 是缩放系数（论文推荐 ~1）。

核心动机是解决**干扰**：不同 task vector 在同一参数上符号相反时会互相抵消，朴素平均会两败俱伤。

### 2.2 ⚠️ 在我们 case 上的关键障碍：LoRA 分解不唯一

TIES 是**逐元素**操作，前提是各 task vector 处在**同一组坐标**下。我们的 residual policy 每层被
分解成 `W = B · A`（如 `fc1_A: [16, 846]`、`fc1_B: [512, 16]`），而这个分解**不唯一**——
对任意可逆 `R`，`(B R⁻¹)(R A)` 给出同一个 `W`。不同 run 会收敛到完全不同的分解。

实测（5 个 object RGB checkpoint，fc1 层，两两对比）：

| pair | cos(A_i, A_j) | cos(B_i, B_j) | cos(W_i, W_j) |
|---|---|---|---|
| 1-6 | 0.1551 | −0.0085 | 0.0011 |
| 1-7 | 0.0113 | 0.0006 | −0.0037 |
| 1-8 | −0.0263 | −0.0010 | 0.0064 |
| 1-9 | −0.0502 | 0.0239 | 0.0120 |
| 6-7 | −0.0339 | −0.0104 | 0.0019 |
| 6-8 | 0.0330 | −0.0090 | −0.0002 |
| 6-9 | −0.0845 | −0.0089 | 0.0009 |
| 7-8 | −0.0069 | 0.0009 | −0.0080 |
| 7-9 | −0.0503 | 0.0007 | −0.0088 |
| 8-9 | −0.0747 | 0.0064 | 0.0089 |

**结论一**：`cos(A_i, A_j)` 全部接近 0，说明各任务的 LoRA 因子处在互不相关的坐标系里。
**因此绝不能把 TIES 直接套在 `A`、`B` 两个矩阵上**——那是在比较无关的坐标，符号选举没有意义。
正确做法是在**有效权重** `W_i = B_i A_i` 上做 TIES，再（若需要）用 SVD 截回 rank-16。

**结论二**：`cos(W_i, W_j)` 也全部接近 0（|cos| < 0.013），即**各任务的有效修正在权重空间中近乎正交**。
这对 TIES 的适用性是个实质性提醒：TIES 的收益来自化解**符号冲突**，而近乎正交意味着本来就几乎没有
冲突，各 task vector 占据彼此独立的子空间。可以预期 TIES 相对朴素平均的增益有限，而两者共同的问题是
**幅度被稀释**（N 个正交块平均后，每块只剩 1/N）——这与正文观察到的「merge-only 明显掉点、需要
post-merge refine」是一致的。这个分析本身就是可以写进 rebuttal 的论点。

**结论三**：`τ_i = θ_i − θ_init` 里的 `θ_init` 在我们这里没有共同基准——每个 run 独立随机初始化
（`_init_lora_weights`：A 用 Kaiming，B 用 `normal_(std=0.01)`）。好在 B 近零初始化 ⇒
初始有效权重 `W_init = B_init A_init ≈ 0`，所以**可以直接取 `τ_i ≈ W_i`**，不需要 `θ_pre`。
这一点让 TIES 在 residual 空间可用；但要在文中说明这个近似及其依据。

### 2.3 与我们自己 merge 算子的关系

值得注意：我们的 `RobustMergeLoRA`（`merge_lora_actors.py`）已经是 TIES 家族的做法——

| 步骤 | TIES | 我们的 RobustMergeLoRA |
|---|---|---|
| 裁剪 | top-K% 幅度保留（K=20%） | `_prune_matrix(A, prune_ratio=0.2)` 裁掉后 20% |
| 补偿 | 无 | `_compute_scaling_matrix` 互补缩放，补回裁剪损失的范数 |
| 符号 | 符号选举 + 只聚合同号项 | **无** |
| 归一化 | 无 | 跨任务把有效权重范数归一到均值 |
| 聚合 | 同号项求均值 | 加权平均 |

所以 TIES 不是一个「完全不同的方法」，而是**我们算子的一个消融**：加上符号选举、去掉范数归一化。
这样对比反而更有说服力——可以逐项说明哪一步带来增益。

---

## 3. 建议的实验方案

按性价比排序。**第 1 项无需任何新训练，用现有 5 个 object RGB checkpoint 就能做。**

### 方案 1（立刻可做）：residual 空间的 merge 算子对比

固定 arrival order `1,6,7,8,9`，固定「全部合并成一个 expert」（去掉 RFC 门控，以隔离算子本身的影响），
比较 4 个融合算子，各出一张 5×5 SR 表：

1. 朴素平均（uniform averaging）
2. RETAIN 线性插值（顺序式，α 扫 {0.1, 0.3, 0.5, 0.7, 0.9}）
3. TIES（在有效权重 `W=B·A` 上，K 扫 {10%, 20%, 50%}，λ=1）
4. 我们的 RobustMergeLoRA（`--restore_norm`，prune_ratio=0.2）

需要新写的代码：TIES 在 `W` 空间的实现 + SVD 截回 rank（约 100 行），RETAIN 插值（约 20 行）。

### 方案 2：门控 × 算子的 2×2

在方案 1 基础上，加上「有无 RFC 门控」这一维，说明增益来自门控还是算子。

### 方案 3（需要你提供资产）：忠实复现 RETAIN

需要每个任务的 VLA SFT checkpoint（或 demo 数据让我自己 SFT）。

---

## 4. 需要你提供 / 确认的

1. **RETAIN 是否要做忠实复现（方案 3）**。若要，需要 5 个 object 任务各自的 VLA SFT checkpoint；
   你提到可以提供 demo 数据或 SFT checkpoint，**直接给 checkpoint 最省事**。
2. **忠实复现时 `θ_pre` 用哪个**。现有 base 已在 object suite 上 SFT 过，用它会让 RETAIN 的卖点
   失效；建议用未经 object 微调的 OpenVLA-OFT 通用 checkpoint，但这会改变整个实验的 base，
   需要你决定是否值得。
3. **TIES 是否接受「在有效权重 `W=B·A` 上做，而非 A/B 因子上做」这一适配**。实测证据见 §2.2，
   我认为这是唯一正确的做法，但它是对原方法的一个调整，需要在 rebuttal 里写明。
4. 是否需要 RETAIN 的 co-FT 和 modality-specific 两个变体（都需要额外资产，我建议**不做**，
   在文中说明理由即可）。

---

## 5. 尚未核实的点

- TIES 官方实现是针对 T5 / IA3 的，`src/ties_merging.py` 的入口耦合了它们的数据与评测栈，
  我打算只复用 `utils/merge_utils.py` 里的三个核心函数（`topk_values_mask` / `resolve_sign` /
  `disjoint_merge`），不跑它的 pipeline。这三个函数是纯张量操作，无外部依赖。
- RETAIN 代码基于 openpi（JAX），我们是 PyTorch，`linear_interpolation` 逻辑极简，直接重写而非移植。
- 两个 baseline 原文都在**全模型权重**上操作，而我们在 residual 上操作，参数量差 4 个数量级
  （residual 约 1.4 M vs VLA 7 B）。TIES 的裁剪比例等超参是在大模型上调的，在小模型上可能需要重新扫。
