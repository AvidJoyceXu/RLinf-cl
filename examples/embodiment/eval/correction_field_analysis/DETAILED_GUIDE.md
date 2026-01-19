# 修正向量场分析工具详细指南

## 目录

1. [概述](#概述)
2. [脚本运行方式](#脚本运行方式)
3. [Python文件详细解析](#python文件详细解析)
4. [配置文件详解](#配置文件详解)
5. [工作流程](#工作流程)
6. [实际案例](#实际案例)

---

## 概述

修正向量场分析工具用于评估两个在不同任务上训练的 LoRA Residual Policy 是否可以安全合并。该工具通过分析两个策略在相同状态空间上的修正向量（residual actions）的相似性，来判断合并是否会导致性能下降或策略冲突。

### 核心概念

- **Residual Policy**: 在 Base Model (OpenVLA-OFT) 基础上添加的修正策略，输出 residual action
- **Final Action**: `final_action = base_action + res_scale * residual_action`
- **Correction Vector Field**: 在不同状态上，residual policy 输出的修正向量构成的向量场
- **方向一致性**: 两个策略在同一状态上输出的修正向量的方向相似度（cosine similarity）
- **尺度一致性**: 两个策略在同一状态上输出的修正向量的幅度差异

---

## 脚本运行方式

### 方式一：使用 Bash 脚本（推荐）

```bash
cd /home/xulingyun/RLinf-cl/examples/embodiment/eval/correction_field_analysis
bash run_correction_field_analysis.sh
```

**Bash 脚本的工作流程：**

1. **读取配置参数**：从脚本顶部的配置区域读取参数
2. **构建 Python 命令**：根据参数构建完整的 Python 命令
3. **执行分析**：调用 Python 脚本执行实际分析
4. **错误处理**：检查退出码并显示成功/失败信息

**配置参数位置：**

```bash
# =============================================================================
# 配置参数区域 - 在这里修改参数
# =============================================================================

CONFIG_FILE="configs/eval_lora_config.yaml"
TASK_I=0
TASK_J=1
CHECKPOINT_I=""
CHECKPOINT_J=""
STATE_METHOD="both_demo"
MAX_DEMOS=5
REFERENCE_TASK="0"
DELTA_DIR=0.5
```

### 方式二：直接使用 Python

```bash
python analyze_correction_vector_field.py \
    --config configs/eval_lora_config.yaml \
    --task_i 0 \
    --task_j 1 \
    --checkpoint_i /path/to/task_i/checkpoint \
    --checkpoint_j /path/to/task_j/checkpoint \
    --state_method both_demo \
    --max_demos 5 \
    --delta_dir 0.5
```

### 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--config` | 是 | Hydra 配置文件路径 |
| `--task_i` | 是 | 任务 i 的 ID |
| `--task_j` | 是 | 任务 j 的 ID |
| `--checkpoint_i` | 否 | 任务 i 的 checkpoint 路径（HuggingFace 格式） |
| `--checkpoint_j` | 否 | 任务 j 的 checkpoint 路径 |
| `--state_method` | 否 | 状态收集方法：`demo`, `both_demo`, `base_rollout` |
| `--max_demos` | 否 | 用于状态收集的 episode/demo 数量（默认：5） |
| `--reference_task` | 否 | 参考任务 ID（仅当 `state_method=demo` 时使用） |
| `--delta_dir` | 否 | 方向一致性危险阈值（默认：0.5） |

---

## Python文件详细解析

### 文件结构

`analyze_correction_vector_field.py` 包含以下主要函数：

1. **配置和环境相关**
   - `load_config()`: 加载 Hydra 配置文件
   - `create_simple_libero_env()`: 创建简化的 LIBERO 环境

2. **状态收集**
   - `collect_state_samples_from_demo()`: 从演示数据收集状态
   - `collect_state_samples_from_base_rollout()`: 从 Base Model rollout 收集状态

3. **模型加载**
   - `load_residual_policy()`: 加载 LoRA Residual Policy
   - `get_eval_action()`: 获取模型的确定性动作（eval 模式）

4. **分析计算**
   - `compute_direction_consistency()`: 计算方向一致性
   - `compute_scale_consistency()`: 计算尺度一致性
   - `pointwise_probe()`: 点对点探测
   - `safety_aggregation()`: 安全聚合分析
   - `merge_decision()`: 合并决策

5. **可视化和输出**
   - `visualize_correction_field_analysis()`: 生成可视化图表
   - `main()`: 主函数，协调整个流程

### 核心函数详解

#### 1. `load_config(config_path)`

**功能**：加载 Hydra 配置文件

**工作流程**：
```python
1. 解析配置文件路径（支持相对路径和绝对路径）
2. 设置 EMBODIED_PATH 环境变量（如果未设置）
3. 使用 Hydra 的 initialize_config_dir 和 compose 加载配置
4. 返回 OmegaConf 配置对象
```

**关键点**：
- 使用 Hydra 的配置系统，支持配置继承和覆盖
- 自动处理配置文件的相对路径
- 设置必要的环境变量以支持配置文件的 `searchpath`

#### 2. `create_simple_libero_env(task_suite_name, task_id, seed=42)`

**功能**：创建简化的 LIBERO 环境用于状态收集

**工作流程**：
```python
1. 获取 benchmark（任务套件）
2. 根据 task_id 获取具体任务
3. 构建 BDDL 文件路径
4. 创建 OffScreenRenderEnv 环境实例
5. 返回环境和任务描述
```

**关键点**：
- 使用 `get_benchmark_overridden()` 获取 benchmark
- 环境是单任务、单环境的简化版本
- 主要用于状态收集，不需要完整的训练环境

#### 3. `collect_state_samples_from_demo(task_suite_name, task_id, env, max_demos=5)`

**功能**：从演示数据中 rollout 收集状态样本

**工作流程**：
```python
1. 获取任务的演示数据路径（HDF5 格式）
2. 加载演示数据文件
3. 遍历每个演示：
   a. 重置环境
   b. 使用演示中的 actions 执行 rollout
   c. 在每个步骤收集 RL 观察（flatten 后的状态）
   d. 保存状态序列
4. 合并所有状态序列
5. 返回状态数组 [N, obs_dim]
```

**状态格式**：
- LIBERO RL 观察：`robot_proprio_state (39-dim) + object_to_robot_relations (35-dim) = 74-dim`
- 每个状态是一个 74 维的向量

**关键点**：
- 使用演示的 actions 执行，确保收集的状态是真实可达的
- 支持限制演示数量以控制计算时间
- 处理环境终止和异常情况

#### 4. `collect_state_samples_from_base_rollout(base_model, task_suite_name, task_id, env, cfg, device, num_episodes=5, max_steps=240)`

**功能**：使用 Base Model 实际 rollout 收集状态样本

**工作流程**：
```python
1. 获取任务描述
2. 对每个 episode：
   a. 重置环境
   b. 在每个步骤：
      - 收集当前 RL 状态
      - 准备 Base Model 输入（图像 + 任务描述）
      - 使用 Base Model 预测 action
      - 执行 action 并更新环境状态
   c. 保存状态序列
3. 合并所有状态序列
4. 返回状态数组 [N, obs_dim]
```

**关键点**：
- **重要**：RL 实际看到的状态是由 Base Model 的行为决定的
- Base Model 是 IL（Imitation Learning）模型，决定了 rollout 的轨迹
- 只有在 Base Model 实际会访问到的状态上，Residual RL 才有修正的机会
- 这种方法更符合实际 RL 训练时的状态分布

**Base Model 输入处理**：
```python
# 图像预处理
agentview_rgb = torch.from_numpy(obs['agentview_rgb']).unsqueeze(0)
eye_in_hand_rgb = torch.from_numpy(obs['eye_in_hand_rgb']).unsqueeze(0)

# 转换为 CHW 格式并归一化
agentview_rgb = agentview_rgb.permute(0, 3, 1, 2).float() / 255.0
eye_in_hand_rgb = eye_in_hand_rgb.permute(0, 3, 1, 2).float() / 255.0

# 构建输入
base_input = {
    'images': {
        'agentview_rgb': agentview_rgb,
        'eye_in_hand_rgb': eye_in_hand_rgb
    },
    'task_descriptions': [task_description],
}
```

#### 5. `load_residual_policy(checkpoint_path, cfg, device)`

**功能**：加载 rlinf LoRA Residual Policy

**工作流程**：
```python
1. 检查 checkpoint 路径类型：
   a. 如果是目录（HuggingFace 格式）：
      - 使用 get_model() 加载模型
      - 模型会自动加载权重
   b. 如果是文件（checkpoint 文件）：
      - 创建模型实例
      - 加载 state_dict
2. 设置模型为 eval 模式
3. 移动到指定设备（CPU/GPU）
4. 返回模型实例
```

**Checkpoint 格式**：
- **HuggingFace 格式**（推荐）：
  - 目录结构：`checkpoint_dir/`
    - `config.json`: 模型配置
    - `pytorch_model.bin` 或 `model.safetensors`: 模型权重
  - 使用 `get_model()` 自动加载

- **Checkpoint 文件格式**：
  - 包含 `state_dict` 或 `model_state_dict` 的字典
  - 需要手动加载权重

#### 6. `get_eval_action(model, states, device)`

**功能**：获取模型的确定性动作（eval 模式）

**工作流程**：
```python
1. 将状态数组转换为 tensor
2. 构建 env_obs 格式：
   {
       'rl_flatten_obs': states_tensor  # [B, obs_dim]
   }
3. 调用 model.predict_action_batch()，mode="eval"
4. 获取 actions: [B, num_action_chunks, action_dim]
5. 取第一个 chunk 的 action: [B, action_dim]
6. 返回 actions
```

**关键点**：
- `mode="eval"` 使用确定性策略（取均值），而不是采样
- 只取第一个 action chunk，因为分析关注的是单步动作
- 批量处理以提高效率

#### 7. `compute_direction_consistency(a1, a2, epsilon=1e-8)`

**功能**：计算两个修正向量的方向一致性（cosine similarity）

**公式**：
```
dir_sim = dot(a1, a2) / (||a1|| * ||a2||)
```

**返回值**：
- `1.0`: 完全同向
- `0.0`: 垂直或一个为零向量
- `-1.0`: 完全反向

**边界情况处理**：
- 两个都是零向量 → 返回 1.0（认为一致）
- 一个是零向量 → 返回 0.0（认为不一致）
- 使用 `np.clip()` 确保结果在 [-1, 1] 范围内

#### 8. `compute_scale_consistency(a1, a2, epsilon=1e-8)`

**功能**：计算两个修正向量的尺度一致性

**公式**：
```
log_scale = |log(||a1|| / ||a2||)|
```

**返回值**：
- `0.0`: 尺度相同
- 值越大，尺度差异越大
- `log_scale = 1.0` 表示约 2.7x 的差异

**边界情况处理**：
- 两个都是零向量 → 返回 0.0
- a2 是零向量 → 返回 `log(||a1|| / epsilon)`（很大的差异）

#### 9. `pointwise_probe(residual1, residual2, states, device)`

**功能**：在状态集合上点对点探测两个 residual policies

**工作流程**：
```python
1. 批量处理状态（batch_size=32）
2. 对每个 batch：
   a. 获取两个策略的 actions
   b. 对每个状态：
      - 计算方向一致性
      - 计算尺度一致性
      - 记录统计信息
3. 返回结果列表，每个元素包含：
   {
       'state_idx': int,
       'state': np.ndarray,  # [obs_dim]
       'a1': np.ndarray,     # [action_dim]
       'a2': np.ndarray,     # [action_dim]
       'dir': float,         # 方向一致性
       'log_scale': float,   # 尺度一致性
       'a1_norm': float,     # a1 的 L2 范数
       'a2_norm': float,     # a2 的 L2 范数
   }
```

**关键点**：
- 批量处理以提高效率
- 记录每个状态的详细信息，用于后续分析和可视化

#### 10. `safety_aggregation(results, delta_dir=0.5)`

**功能**：安全聚合分析，计算统计信息

**计算的统计量**：

1. **方向一致性统计**：
   - `mean_dir`: 平均方向一致性
   - `std_dir`: 标准差
   - `median_dir`: 中位数
   - `min_dir`, `max_dir`: 最小值和最大值

2. **尺度一致性统计**：
   - `mean_log_scale`: 平均 log scale
   - `std_log_scale`: 标准差
   - `median_log_scale`: 中位数
   - `min_log_scale`, `max_log_scale`: 最小值和最大值

3. **最坏情况分析**：
   - `dangerous_mask`: 布尔数组，标记危险状态（dir < delta_dir）
   - `dangerous_count`: 危险状态数量
   - `p_bad`: 危险状态比例
   - `dangerous_mean_dir`: 危险状态的平均方向一致性
   - `dangerous_mean_log_scale`: 危险状态的平均尺度一致性

**返回值**：
包含所有统计信息的字典，以及原始数据（用于可视化）

#### 11. `merge_decision(aggregation, thresholds=None)`

**功能**：基于聚合结果做出合并决策

**决策规则**：

默认阈值：
```python
thresholds = {
    'threshold_dir': 0.7,      # 整体方向一致性阈值
    'threshold_scale': 1.0,    # log scale 阈值（约 2.7x 差异）
    'threshold_bad': 0.05,     # 危险状态比例阈值（5%）
}
```

**决策逻辑**：
```python
can_merge = True
reasons = []

if mean_dir < threshold_dir:
    can_merge = False
    reasons.append("Overall direction misalignment")

if mean_log_scale > threshold_scale:
    can_merge = False
    reasons.append("Scale mismatch too large")

if p_bad > threshold_bad:
    can_merge = False
    reasons.append("Too many dangerous states")
```

**返回值**：
```python
{
    'can_merge': bool,
    'reasons': List[str],
    'mean_dir': float,
    'mean_log_scale': float,
    'p_bad': float,
    'thresholds': dict,
}
```

#### 12. `visualize_correction_field_analysis(results, aggregation, decision, output_dir, task_i, task_j)`

**功能**：生成可视化图表

**生成的图表**（2x2 布局）：

1. **方向一致性分布**（左上）：
   - 直方图显示方向一致性的分布
   - 标记平均值和危险阈值

2. **尺度一致性分布**（右上）：
   - 直方图显示 log scale 的分布
   - 标记平均值

3. **方向 vs 尺度散点图**（左下）：
   - 绿色点：安全状态
   - 红色点：危险状态
   - 垂直虚线：危险阈值

4. **决策结果总结**（右下）：
   - 合并决策（✅ CAN MERGE / ❌ CANNOT MERGE）
   - 统计信息
   - 阈值设置
   - 决策原因

**保存位置**：
`{output_dir}/correction_field_analysis_task{i}_task{j}.png`

#### 13. `main()`

**功能**：主函数，协调整个分析流程

**完整流程**：

```python
1. 解析命令行参数
2. 加载配置文件
3. 设置设备（CPU/GPU）
4. 创建输出目录
5. 加载 Base Model
6. 根据 state_method 收集状态：
   - demo: 从单个参考任务的演示数据收集
   - both_demo: 从两个任务的演示数据收集并合并
   - base_rollout: 从 Base Model rollout 收集（推荐）
7. 加载两个 Residual Policies
8. 点对点探测（pointwise_probe）
9. 安全聚合分析（safety_aggregation）
10. 合并决策（merge_decision）
11. 可视化（visualize_correction_field_analysis）
12. 保存分析摘要（summary_analysis.txt）
13. 打印结果
```

---

## 配置文件详解

### 配置文件结构

`configs/eval_lora_config.yaml` 使用 Hydra 配置系统，支持配置继承和覆盖。

### 关键配置项

#### 1. Hydra 配置

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/
```

**作用**：
- `run.dir`: Hydra 运行的工作目录
- `output_subdir`: 输出子目录（null 表示不创建子目录）
- `searchpath`: 配置文件的搜索路径，使用环境变量 `EMBODIED_PATH`

#### 2. 默认配置继承

```yaml
defaults:
  - env/libero_spatial_task0@env.train
  - env/libero_spatial_task0@env.eval
  - model/lora_residual_policy@actor.model
  - model/openvla_oft@actor.base_model
  - training_backend/fsdp@actor.fsdp_config
```

**作用**：
- 从其他配置文件继承配置
- `@` 符号指定配置的目标位置
- 例如：`env/libero_spatial_task0@env.train` 表示从 `env/libero_spatial_task0.yaml` 加载配置，并放在 `env.train` 下

#### 3. Runner 配置

```yaml
runner:
  eval_policy_path: ""  # 默认 checkpoint 路径
  task_type: embodied
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "correction_field_analysis"
    logger_backends: []
```

**作用**：
- `eval_policy_path`: 默认的 checkpoint 路径（可通过命令行参数覆盖）
- `logger.log_path`: 结果保存路径
- `logger.logger_backends`: 日志后端（空列表表示不记录日志）

**影响**：
- 如果未指定 `--checkpoint_i` 或 `--checkpoint_j`，将使用 `eval_policy_path`
- 分析结果保存在 `{log_path}/correction_field_analysis_task{i}_task{j}_{timestamp}/`

#### 4. Residual Policy 配置

```yaml
residual_policy:
  enabled: True
  base_model_path: "/path/to/openvla_oft/model"
  res_scale: 0.05
  prog_explore: 500
  prog_explore_threshold: 100
```

**作用**：
- `base_model_path`: Base Model (OpenVLA-OFT) 的路径
- `res_scale`: Residual action 的缩放因子
- `prog_explore`: 渐进探索步数（分析时不影响）

**影响**：
- Base Model 路径用于 `collect_state_samples_from_base_rollout()`
- 如果 `state_method=base_rollout`，必须正确设置此路径

#### 5. Network 配置

```yaml
network:
  actor_input: "obs"  # ["obs", "obs_base_action"]
  critic_input: "sum"  # ["res", "sum", "concat"]
```

**作用**：
- `actor_input`: Actor 的输入模式
  - `"obs"`: 仅观察
  - `"obs_base_action"`: 观察 + base action
- `critic_input`: Critic 的输入模式（分析时不影响）

**影响**：
- 影响 `get_eval_action()` 中如何构建模型输入
- 如果 `actor_input="obs_base_action"`，需要提供 base action（但在分析中我们只使用 obs）

#### 6. Actor 模型配置

```yaml
actor:
  model:
    model_name: "lora_residual_policy"
    action_dim: 7
    num_action_chunks: 8
    precision: "bf16"
    obs_dim: 74
    actor_input: ${network.actor_input}
```

**作用**：
- `model_name`: 模型类型（`lora_residual_policy`）
- `action_dim`: 动作维度（LIBERO 是 7）
- `num_action_chunks`: Action chunks 数量
- `obs_dim`: 观察维度（LIBERO RL obs 是 74）
- `actor_input`: 从 `network.actor_input` 继承

**影响**：
- 用于创建模型实例
- `obs_dim` 必须与实际观察维度匹配（74）
- `action_dim` 必须与实际动作维度匹配（7）

#### 7. Base Model 配置

```yaml
actor:
  base_model:
    model_type: "openvla_oft"
    model_path: "/path/to/openvla_oft/model"
    precision: ${actor.model.precision}
    unnorm_key: libero_spatial_no_noops
    max_prompt_length: 128
```

**作用**：
- `model_type`: Base Model 类型（`openvla_oft`）
- `model_path`: Base Model 路径
- `unnorm_key`: 动作归一化键
- `max_prompt_length`: 最大 prompt 长度

**影响**：
- 用于加载 Base Model
- 如果 `state_method=base_rollout`，必须正确设置此配置

#### 8. 环境配置

```yaml
env:
  train:
    total_num_envs: 1
    max_episode_steps: 240
    task_suite_name: libero_spatial
```

**作用**：
- `total_num_envs`: 环境数量（分析时只需要 1 个）
- `max_episode_steps`: 最大 episode 步数
- `task_suite_name`: 任务套件名称

**影响**：
- `task_suite_name` 用于确定使用哪个 benchmark
- `max_episode_steps` 用于限制 rollout 的最大步数

### 配置文件如何影响运行

1. **模型加载**：
   - `actor.model.*` → 用于创建 Residual Policy 模型
   - `actor.base_model.*` → 用于加载 Base Model

2. **状态收集**：
   - `env.train.task_suite_name` → 确定任务套件
   - `residual_policy.base_model_path` → Base Model 路径（用于 base_rollout）

3. **结果保存**：
   - `runner.logger.log_path` → 结果保存路径

4. **模型行为**：
   - `network.actor_input` → 影响模型输入格式
   - `actor.model.obs_dim` → 必须与实际观察维度匹配

---

## 工作流程

### 完整流程图

```
开始
  ↓
解析命令行参数
  ↓
加载配置文件（Hydra）
  ↓
设置设备（CPU/GPU）
  ↓
创建输出目录
  ↓
加载 Base Model
  ↓
根据 state_method 收集状态：
  ├─ demo → collect_state_samples_from_demo()
  ├─ both_demo → collect_state_samples_from_demo() × 2
  └─ base_rollout → collect_state_samples_from_base_rollout() × 2
  ↓
加载两个 Residual Policies
  ↓
点对点探测（pointwise_probe）
  ├─ 批量处理状态
  ├─ 获取两个策略的 actions
  └─ 计算方向一致性和尺度一致性
  ↓
安全聚合分析（safety_aggregation）
  ├─ 计算统计信息
  └─ 识别危险状态
  ↓
合并决策（merge_decision）
  ├─ 检查方向一致性阈值
  ├─ 检查尺度一致性阈值
  └─ 检查危险状态比例
  ↓
可视化（visualize_correction_field_analysis）
  ├─ 生成方向一致性分布图
  ├─ 生成尺度一致性分布图
  ├─ 生成散点图
  └─ 生成决策总结
  ↓
保存分析摘要（summary_analysis.txt）
  ↓
结束
```

### 状态收集方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| `demo` | 快速，使用真实演示数据 | 可能不反映 Base Model 的实际行为 | 快速初步分析 |
| `both_demo` | 覆盖两个任务的状态空间 | 可能包含 Base Model 不会访问的状态 | 任务特定的分析 |
| `base_rollout` | **最符合实际 RL 训练时的状态分布** | 较慢，需要运行 Base Model | **推荐用于最终分析** |

### 分析指标解释

1. **方向一致性（Direction Consistency）**：
   - 衡量两个策略在同一状态上输出的修正向量是否同向
   - 值域：[-1, 1]
   - `1.0`: 完全同向（理想）
   - `0.0`: 垂直或一个为零向量
   - `-1.0`: 完全反向（危险）

2. **尺度一致性（Scale Consistency）**：
   - 衡量两个策略在同一状态上输出的修正向量的幅度差异
   - 值域：[0, +∞)
   - `0.0`: 幅度相同（理想）
   - `1.0`: 约 2.7x 差异
   - 值越大，差异越大

3. **危险状态（Dangerous States）**：
   - 方向一致性 < `delta_dir` 的状态
   - 这些状态上两个策略的修正方向差异较大，合并可能导致冲突

---

## 实际案例

### 案例 1：分析任务 0 和任务 1

**场景**：已经训练了任务 0 和任务 1 的 LoRA Residual Policies，想判断是否可以合并。

**步骤**：

1. **准备 checkpoint 路径**：
   ```bash
   CHECKPOINT_I="/workspace/RLinf/logs/.../task0/checkpoints/global_step_9000/actor/huggingface_model"
   CHECKPOINT_J="/workspace/RLinf/logs/.../task1/checkpoints/global_step_9000/actor/huggingface_model"
   ```

2. **修改脚本参数**：
   ```bash
   TASK_I=0
   TASK_J=1
   STATE_METHOD="base_rollout"  # 推荐使用 base_rollout
   MAX_DEMOS=5
   DELTA_DIR=0.5
   ```

3. **运行分析**：
   ```bash
   bash run_correction_field_analysis.sh
   ```

4. **查看结果**：
   - 可视化图表：`results/correction_field_analysis_task0_task1_*/correction_field_analysis_task0_task1.png`
   - 分析摘要：`results/correction_field_analysis_task0_task1_*/summary_analysis.txt`

5. **解读结果**：
   - 如果 `can_merge=True`，可以尝试合并
   - 如果 `can_merge=False`，查看 `reasons` 了解原因

### 案例 2：快速初步分析

**场景**：想快速了解两个策略的相似性，不需要精确分析。

**步骤**：

1. **使用 demo 方法**：
   ```bash
   STATE_METHOD="demo"
   MAX_DEMOS=3  # 减少 demo 数量以加快速度
   ```

2. **运行分析**：
   ```bash
   python analyze_correction_vector_field.py \
       --config configs/eval_lora_config.yaml \
       --task_i 0 \
       --task_j 1 \
       --checkpoint_i /path/to/task0/checkpoint \
       --checkpoint_j /path/to/task1/checkpoint \
       --state_method demo \
       --max_demos 3
   ```

### 案例 3：调整阈值

**场景**：默认阈值太严格，想放宽合并条件。

**方法**：修改 `merge_decision()` 函数中的阈值：

```python
thresholds = {
    'threshold_dir': 0.6,      # 从 0.7 降低到 0.6
    'threshold_scale': 1.5,    # 从 1.0 增加到 1.5
    'threshold_bad': 0.10,     # 从 0.05 增加到 0.10
}
```

或者在命令行中调整 `--delta_dir`：

```bash
--delta_dir 0.4  # 从默认 0.5 降低到 0.4
```

---

## 常见问题

### Q1: 为什么推荐使用 `base_rollout` 方法？

**A**: 因为 RL 实际看到的状态是由 Base Model 的行为决定的。只有在 Base Model 实际会访问到的状态上，Residual RL 才有修正的机会。使用 `base_rollout` 可以更准确地反映实际训练时的状态分布。

### Q2: Checkpoint 路径应该指向哪里？

**A**: 应该指向 HuggingFace 格式的模型目录，包含：
- `config.json`
- `pytorch_model.bin` 或 `model.safetensors`

例如：
```
/workspace/RLinf/logs/.../checkpoints/global_step_9000/actor/huggingface_model/
├── config.json
├── pytorch_model.bin
└── ...
```

### Q3: 如何理解合并决策的结果？

**A**: 
- `can_merge=True`: 两个策略的修正向量场相似度较高，可以尝试合并
- `can_merge=False`: 查看 `reasons` 了解具体原因：
  - 方向不一致：两个策略在相同状态上的修正方向差异较大
  - 尺度不匹配：两个策略的修正幅度差异较大
  - 危险状态过多：有太多状态上两个策略的修正方向差异较大

### Q4: 分析需要多长时间？

**A**: 取决于：
- 状态收集方法：`base_rollout` 最慢，`demo` 最快
- `max_demos` 数量：越多越慢，但结果更准确
- GPU 可用性：有 GPU 会快很多

典型时间：
- `demo` + `max_demos=5`: 约 1-2 分钟
- `base_rollout` + `max_demos=5`: 约 5-10 分钟

### Q5: 如何提高分析准确性？

**A**:
1. 使用 `base_rollout` 方法（最准确）
2. 增加 `max_demos` 数量（更多状态样本）
3. 确保 Base Model 路径正确
4. 确保 checkpoint 路径正确且模型已训练充分

---

## 总结

修正向量场分析工具通过系统化的方法评估两个 LoRA Residual Policies 是否可以安全合并。核心思想是：

1. **收集状态样本**：在真实的状态空间上收集样本（推荐使用 Base Model rollout）
2. **点对点探测**：比较两个策略在每个状态上的输出
3. **统计分析**：计算方向一致性和尺度一致性
4. **安全决策**：基于阈值判断是否可以合并

该工具为持续学习中的模型合并提供了量化的评估方法，帮助研究者做出更明智的决策。

