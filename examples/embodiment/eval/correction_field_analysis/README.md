# Correction Vector Field Analysis for rlinf LoRA Residual Policy

这个工具用于分析两个 rlinf LoRA residual policies 的修正向量场相似性，判断是否可以合并（merge）。

## 功能说明

该工具实现了以下功能：

1. **状态收集**：支持三种方式收集状态样本
   - `demo`: 从单个参考任务的演示数据收集
   - `both_demo`: 从两个任务的演示数据收集并合并
   - `base_rollout`: 从 Base Model rollout 收集（推荐，更符合实际RL看到的states）

2. **点对点探测**：在收集的状态上比较两个 residual policies 的输出

3. **安全聚合分析**：计算方向一致性和尺度一致性统计

4. **合并决策**：基于阈值判断是否可以合并两个 policies

5. **可视化**：生成分析结果的可视化图表

## 文件结构

```
correction_field_analysis/
├── analyze_correction_vector_field.py  # 主分析脚本
├── run_correction_field_analysis.sh     # Bash启动脚本
├── configs/
│   └── eval_lora_config.yaml            # 评估配置文件示例
└── README.md                            # 本文件
```

## 使用方法

### 1. 准备配置文件

首先，确保配置文件 `configs/eval_lora_config.yaml` 中的路径正确：
- `residual_policy.base_model_path`: Base Model (OpenVLA-OFT) 的路径
- `actor.base_model.model_path`: Base Model 的路径（同上）
- `runner.eval_policy_path`: 默认的 checkpoint 路径（可选，可通过命令行参数覆盖）

### 2. 修改启动脚本参数

编辑 `run_correction_field_analysis.sh`，修改以下参数：

```bash
# 任务配置
TASK_I=0                                      # 任务i ID
TASK_J=1                                      # 任务j ID

# Checkpoint路径配置
CHECKPOINT_I="/path/to/task_i/checkpoint"    # 任务i的checkpoint路径（huggingface model目录）
CHECKPOINT_J="/path/to/task_j/checkpoint"    # 任务j的checkpoint路径（huggingface model目录）

# 状态收集配置
STATE_METHOD="both_demo"                      # 状态收集方法: demo, both_demo, base_rollout
MAX_DEMOS=5                                   # 用于状态收集的episode/demo数量
REFERENCE_TASK="0"                            # 参考任务ID（仅当state_method=demo时使用）

# 分析阈值配置
DELTA_DIR=0.5                                 # 方向一致性危险阈值（0-1之间）
```

### 3. 运行分析

```bash
cd /home/xulingyun/RLinf-cl/examples/embodiment/eval/correction_field_analysis
bash run_correction_field_analysis.sh
```

或者直接使用 Python：

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

## 参数说明

### 必需参数

- `--config`: 评估配置文件路径
- `--task_i`: 任务 i 的 ID
- `--task_j`: 任务 j 的 ID

### 可选参数

- `--checkpoint_i`: 任务 i 的 checkpoint 路径（huggingface model 目录）。如果未指定，将使用配置文件中的 `runner.eval_policy_path`
- `--checkpoint_j`: 任务 j 的 checkpoint 路径。如果未指定，将使用与 checkpoint_i 相同的路径
- `--state_method`: 状态收集方法
  - `demo`: 从单个参考任务的演示数据收集
  - `both_demo`: 从两个任务的演示数据收集并合并
  - `base_rollout`: 从 Base Model rollout 收集（推荐）
- `--max_demos`: 用于状态收集的 episode/demo 数量（默认：5）
- `--reference_task`: 参考任务 ID（仅当 `state_method=demo` 时使用）
- `--delta_dir`: 方向一致性危险阈值，0-1 之间（默认：0.5）

## 输出结果

分析完成后，会在结果目录中生成：

1. **可视化图表** (`correction_field_analysis_task{i}_task{j}.png`):
   - 方向一致性分布
   - 尺度一致性分布
   - 方向 vs 尺度散点图
   - 合并决策总结

2. **分析摘要** (`summary_analysis.txt`):
   - 任务对信息
   - Checkpoint 路径
   - 状态收集统计
   - 方向一致性统计
   - 尺度一致性统计
   - 最坏情况分析
   - 合并决策和原因

结果目录路径：`{runner.logger.log_path}/correction_field_analysis_task{i}_task{j}_{timestamp}/`

## 合并决策阈值

默认阈值（可在代码中修改）：

- `threshold_dir`: 0.7（整体方向一致性阈值）
- `threshold_scale`: 1.0（log scale 阈值，约 2.7x 差异）
- `threshold_bad`: 0.05（危险 state 比例阈值，5%）

## 注意事项

1. **Checkpoint 格式**：rlinf 的 checkpoint 是 HuggingFace 格式（包含 `config.json` 和模型权重文件）

2. **Base Model**：需要确保 Base Model (OpenVLA-OFT) 路径正确

3. **环境依赖**：需要安装 rlinf 和相关依赖，包括：
   - libero
   - rlinf
   - torch
   - numpy
   - matplotlib
   - seaborn
   - hydra-core
   - omegaconf

4. **GPU 使用**：如果有 GPU，脚本会自动使用 GPU 加速

## 示例

分析任务 0 和任务 1 的 LoRA residual policies：

```bash
python analyze_correction_vector_field.py \
    --config configs/eval_lora_config.yaml \
    --task_i 0 \
    --task_j 1 \
    --checkpoint_i /workspace/RLinf/logs/.../task0/checkpoints/global_step_9000/actor/huggingface_model \
    --checkpoint_j /workspace/RLinf/logs/.../task1/checkpoints/global_step_9000/actor/huggingface_model \
    --state_method base_rollout \
    --max_demos 5 \
    --delta_dir 0.5
```

## 参考

该工具基于 `libero-sac` 文件夹下的 `analyze_correction_vector_field.py` 实现，适配了 rlinf 的架构：
- 使用 rlinf 的 `LoRAResidualPolicy` 模型
- 使用 rlinf 的环境和配置系统（Hydra）
- 使用 rlinf 的 checkpoint 格式（HuggingFace）

