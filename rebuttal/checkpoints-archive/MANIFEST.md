# RGB residual checkpoints — HuggingFace 存档清单

25 个 checkpoint，共 78 MB。全部是 **RGB observation 模式**的 LoRA residual policy
（`obs_dim = 846 = 2×384 冻结 DINOv2-small CLS + 2×39 proprio`）。

⚠️ **不能与 privileged 模式的 checkpoint 互换**：`fc1_A` 输入宽度是 846，privileged 版是 88（object）/ 74（spatial），加载会直接报形状错。

⚠️ 加载时必须真正读入权重。`rlinf/models/embodiment/residual_policy/get_model()` 在
`model_path` 为空时只构造随机初始化的模块；本仓库 commit `cd7d6b7` 之前它**完全忽略**
`model_path`，任何依赖它加载的分析都是在随机权重上跑的。

## 基座模型

| suite | base policy |
|---|---|
| libero_object | `Haozhan72/Openvla-oft-SFT-libero-object-traj1` (snapshot `62e5a8da…`) |
| libero_spatial | `Haozhan72/Openvla-oft-SFT-libero-spatial-traj1` (snapshot `39e5240e…`) |
| 视觉编码器（两者共用） | `facebook/dinov2-small`，冻结，CLS pooling，224×224 |

residual 以 `a = π_base(o) + α·tanh(u)`、`α = res_scale = 0.05` 作用于 base，
**跨 base 的 residual 合并在数学上没有定义**，object 与 spatial 的 checkpoint 不可互相合并。

---

## 1. `expert/` — 单任务专家（10 个）

训练 1000 个 global step；`save_interval=500`，正文统一用 **step 1000**。
SR 为该专家在自身任务上的成功率，评测协议见文末。

| 目录 | suite / task | SR |
|---|---|---|
| `libero_object_task1` | object 1 | 0.719 |
| `libero_object_task6` | object 6 | 0.844 |
| `libero_object_task7` | object 7 | 0.781 |
| `libero_object_task8` | object 8 | 0.969 |
| `libero_object_task9` | object 9 | 0.688 |
| `libero_spatial_task0` | spatial 0 | 0.938 |
| `libero_spatial_task2` | spatial 2 | 1.000 |
| `libero_spatial_task3` | spatial 3 | 0.781 |
| `libero_spatial_task6` | spatial 6 | 1.000 |
| `libero_spatial_task7` | spatial 7 | 1.000 |

object 平均 0.800、spatial 平均 0.944。frozen base 对照（object）：0.188 / 0.844 / 0.250 / 0.438 / 0.375，均值 0.419。

## 2. `merged/` — RFC 门控下的合并专家（10 个）

由 `toolkits/merge_lora_policies/quick_merge.py --restore_norm` **增量**生成
（把新任务并进已选中的 expert，而不是多个单任务一次性平均），对应 τ sweep 的各个分支。
全部是 **merge-only，未经 refine**。

| 目录 | 覆盖任务 | 出现于 τ | 在覆盖任务上的 SR |
|---|---|---|---|
| `libero_object_task6_8` | 6, 8 | 0.20 | 0.031 / 0.312 |
| `libero_object_task1_6` | 1, 6 | 0.16 | 0.031 / 0.188 |
| `libero_object_task1_6_8` | 1, 6, 8 | 0.10 | 0.000 / 0.781 / 0.438 |
| `libero_object_task1_6_8_9` | 1, 6, 8, 9 | 0.05, 0.0 | 0.000 / 0.281 / 0.375 / 0.688 |
| `libero_spatial_task6_7` | 6, 7 | 0.20, 0.15 | 0.812 / 0.312 |
| `libero_spatial_task0_6` | 0, 6 | 中间产物 | 0.688 / 0.344 |
| `libero_spatial_task0_6_7` | 0, 6, 7 | 0.10 | 0.906 / 1.000 / 0.719 |
| `libero_spatial_task0_3` | 0, 3 | 0.05 | 0.156 / 0.375 |
| `libero_spatial_task0_3_6` | 0, 3, 6 | 0.0 | 0.281 / 0.406 / 0.312 |
| `libero_spatial_task0_3_6_7` | 0, 3, 6, 7 | 0.0 | — |

`libero_spatial_task0_6_7` 是全部结果里最好的合并专家：把 5 个专家压成 3 个，
bank `0-6-7 | 2 | 3` 的 Merge-SR 达到 **0.881**（不合并是 0.944），且零额外环境交互。

## 3. `baseline-merge/` — 其他合并算子（4 个）

固定其余环节，只替换合并算子，用于 AC.K4 的 parameter-space merging 对照。

| 目录 | 算子 | 覆盖任务上的均值 |
|---|---|---|
| `uniform_libero_spatial_task6_7` | 逐元素平均 A、B 因子 | **0.844** |
| `ties_libero_spatial_task6_7` | TIES（K=0.2, λ=1，作用于 W=BA 后 SVD 回投 rank-16） | 0.594 |
| `uniform_libero_object_task6_8` | 逐元素平均 | **0.344** |
| `ties_libero_object_task6_8` | TIES | 0.281 |

⚠️ 两个 suite 上排序一致：**uniform averaging > TIES > 我们的 RobustMergeLoRA**
（对应 `merged/` 里的 0.562 与 0.172）。详见 `rebuttal/results-rgb-merge.md` §6c。

## 4. `refined/` — merge 后 refine（1 个）

| 目录 | 说明 |
|---|---|
| `libero_spatial_task6_7_refine300` | 从 `merged/libero_spatial_task6_7` 热启动，在任务 6、7 上再训 300 步 |

交互成本实测：153,600 环境步 / 640 rollouts / 约 2.8 h（两个任务合计）。
训练内 eval 为 0.906@99、0.875@199，**离线复评尚未完成**——训练内 eval 实测系统性高于离线评测 0.03–0.27，
不可直接引用。

libero_object 的对应实验（`merged/libero_object_task6_8` 上 refine）**连续三次失败**并已放弃，故无此 checkpoint。

---

## 评测协议（所有 SR 数字共用）

`env.eval.max_trials_per_task: 1` + `use_fixed_reset_state_ids: True` + `total_num_envs: 32`
⇒ **每个任务只用一个固定初始状态**，32 个 env 跑同一场景，SR 的分母是 32 次 rollout。
方差只来自 base policy 的解码采样（`do_sample: True`、`temperature_eval: 0.6`），
不来自初始状态多样性。p ≈ 0.3 时二项标准误约 0.08；同一 checkpoint 重复四次实测为
0.688 / 0.781 / 0.781 / 0.688。

⚠️ 与 openvla-oft 官方脚本（每任务 50 个不同初始状态）的数字**不可直接比较**。

## 复现

见仓库 `README.md` 的 “CoRL 2026 rebuttal experiments” 一节，
结果与分析见 `rebuttal/results-rgb-merge.md`。
