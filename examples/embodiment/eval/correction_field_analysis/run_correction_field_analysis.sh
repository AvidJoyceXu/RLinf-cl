#!/bin/bash

###############################################################################
# Correction Vector Field Analysis Runner for rlinf LoRA Residual Policy
# 
# 使用方法:
#   bash run_correction_field_analysis.sh
# 
# 或者修改下面的配置参数后运行:
#   bash run_correction_field_analysis.sh
###############################################################################

# =============================================================================
# 配置参数区域 - 在这里修改参数
# =============================================================================

# 基础配置
CONFIG_FILE="configs/eval_lora_config.yaml"  # 评估配置文件路径

# 任务配置
TASK_I=0                                      # 任务i ID
TASK_J=3                                      # 任务j ID

# Checkpoint路径配置
# 如果为空，将使用config文件中的eval_policy_path
CHECKPOINT_I="/workspace/RLinf/logs/20260117-15:06:53-libero_spatial_task0_lora_residual_sac_openvlaoft/libero_spatial_task0_rand_trials_lora_residual_sac_openvlaoft/checkpoints/global_step_9000/actor/huggingface_model"                               # 任务i的checkpoint路径（可选，huggingface model目录）
CHECKPOINT_J="/workspace/RLinf/logs/20260117-15:07:09-libero_spatial_task3_lora_residual_sac_openvlaoft/libero_spatial_task3_rand_trials_lora_residual_sac_openvlaoft/checkpoints/global_step_4000/actor/huggingface_model"                               # 任务j的checkpoint路径（可选，huggingface model目录）

# 状态收集配置
STATE_METHOD="base_rollout"                      # 状态收集方法: demo, both_demo, base_rollout
MAX_DEMOS=5                                   # 用于状态收集的episode/demo数量
REFERENCE_TASK="0"                            # 参考任务ID（仅当state_method=demo时使用）

# 分析阈值配置
DELTA_DIR=0.5                                 # 方向一致性危险阈值（0-1之间）

# =============================================================================
# 脚本执行区域 - 通常不需要修改
# =============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "Correction Vector Field Analysis (rlinf LoRA Residual Policy)"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Config File:      $CONFIG_FILE"
echo "  Task I:           $TASK_I"
echo "  Task J:           $TASK_J"
echo "  Checkpoint I:     ${CHECKPOINT_I:-'(using config default)'}"
echo "  Checkpoint J:     ${CHECKPOINT_J:-'(using config default)'}"
echo "  State Method:     $STATE_METHOD"
echo "  Max Demos:        $MAX_DEMOS"
echo "  Reference Task:   ${REFERENCE_TASK:-'(auto)'}"
echo "  Delta Dir:        $DELTA_DIR"
echo ""

# 构建Python命令
PYTHON_CMD="python analyze_correction_vector_field.py"
PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
PYTHON_CMD="$PYTHON_CMD --task_i $TASK_I"
PYTHON_CMD="$PYTHON_CMD --task_j $TASK_J"
PYTHON_CMD="$PYTHON_CMD --state_method $STATE_METHOD"
PYTHON_CMD="$PYTHON_CMD --max_demos $MAX_DEMOS"
PYTHON_CMD="$PYTHON_CMD --delta_dir $DELTA_DIR"

# 添加可选的checkpoint路径
if [ -n "$CHECKPOINT_I" ]; then
    PYTHON_CMD="$PYTHON_CMD --checkpoint_i $CHECKPOINT_I"
fi

if [ -n "$CHECKPOINT_J" ]; then
    PYTHON_CMD="$PYTHON_CMD --checkpoint_j $CHECKPOINT_J"
fi

# 添加可选的reference_task（仅当state_method=demo时使用）
if [ -n "$REFERENCE_TASK" ]; then
    PYTHON_CMD="$PYTHON_CMD --reference_task $REFERENCE_TASK"
fi

echo "Running command:"
echo "  $PYTHON_CMD"
echo ""
echo "=============================================================================="
echo ""

# 执行Python脚本
eval $PYTHON_CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✅ Analysis completed successfully!"
    echo "=============================================================================="
else
    echo ""
    echo "=============================================================================="
    echo "❌ Analysis failed with exit code: $EXIT_CODE"
    echo "=============================================================================="
    exit $EXIT_CODE
fi

