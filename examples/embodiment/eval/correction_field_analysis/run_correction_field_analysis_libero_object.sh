#!/bin/bash

###############################################################################
# Batch Correction Vector Field Analysis Runner for rlinf LoRA Residual Policy
# 
# 使用方法:
#   bash run_correction_field_analysis.sh
# 
# 批量分析多个task pairs的correction field，跳过指定的task IDs
###############################################################################

# =============================================================================
# 配置参数区域 - 在这里修改参数
# =============================================================================

# 基础配置
CONFIG_FILE="configs/eval_lora_config.yaml"  # 评估配置文件路径

# Checkpoint路径配置 - 为每个task指定checkpoint路径
# 格式：task_id:checkpoint_path
# 如果某个task的checkpoint为空，将使用config文件中的eval_policy_path
declare -A CHECKPOINT_PATHS
# CHECKPOINT_PATHS[0]="/workspace/RLinf/logs/20260121-07:24:09-libero_spatial_task0_lora_residual_sac_openvlaoft/task0_single_trial_lora_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
# CHECKPOINT_PATHS[2]="/workspace/RLinf/logs/20260119-10:05:34-libero_spatial_task2_lora_residual_sac_openvlaoft/task2_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"  # 如果为空，将使用config默认路径
# CHECKPOINT_PATHS[3]="/workspace/RLinf/logs/20260119-13:18:01-libero_spatial_task3_lora_residual_sac_openvlaoft/task3_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
# CHECKPOINT_PATHS[4]="/workspace/RLinf/logs/20260119-16:57:44-libero_spatial_task4_lora_residual_sac_openvlaoft/task4_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"  # 如果为空，将使用config默认路径
# CHECKPOINT_PATHS[6]="/workspace/RLinf/logs/20260120-02:34:04-libero_spatial_task6_lora_residual_sac_openvlaoft/task6_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"  # 如果为空，将使用config默认路径
# CHECKPOINT_PATHS[7]="/workspace/RLinf/logs/20260120-02:34:18-libero_spatial_task7_lora_residual_sac_openvlaoft/task7_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"  # 如果为空，将使用config默认路径

# bash examples/embodiment/batch_eval_cross_task.sh libero_object_task0_lora_residual_sac_openvlaoft \
#     /skip \
#     /workspace/RLinf/logs/20260122-13:30:27-libero_object_task1_lora_residual_sac_openvlaoft.yaml/libero_object_task1_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#     /workspace/RLinf/logs/20260122-13:09:42-libero_object_task2_lora_residual_sac_openvlaoft.yaml/libero_object_task2_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#     /skip \
#     /skip \
#     /skip \
#     /workspace/RLinf/logs/20260123-03:15:44-libero_object_task6_lora_residual_sac_openvlaoft.yaml/libero_object_task6_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#     /workspace/RLinf/logs/20260123-04:18:12-libero_object_task7_lora_residual_sac_openvlaoft.yaml/libero_object_task7_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#     /workspace/RLinf/logs/20260123-04:18:20-libero_object_task8_lora_residual_sac_openvlaoft.yaml/libero_object_task8_single_trial_lora_openvlaoft/checkpoints/global_step_2000 \
#     /workspace/RLinf/logs/20260123-11:48:31-libero_object_task9_lora_residual_sac_openvlaoft.yaml/libero_object_task9_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model

CHECKPOINT_PATHS[1]="/workspace/RLinf/logs/20260122-13:30:27-libero_object_task1_lora_residual_sac_openvlaoft.yaml/libero_object_task1_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
CHECKPOINT_PATHS[2]="/workspace/RLinf/logs/20260122-13:09:42-libero_object_task2_lora_residual_sac_openvlaoft.yaml/libero_object_task2_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
CHECKPOINT_PATHS[6]="/workspace/RLinf/logs/20260123-03:15:44-libero_object_task6_lora_residual_sac_openvlaoft.yaml/libero_object_task6_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
CHECKPOINT_PATHS[7]="/workspace/RLinf/logs/20260123-04:18:12-libero_object_task7_lora_residual_sac_openvlaoft.yaml/libero_object_task7_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
CHECKPOINT_PATHS[8]="/workspace/RLinf/logs/20260123-04:18:20-libero_object_task8_lora_residual_sac_openvlaoft.yaml/libero_object_task8_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"
CHECKPOINT_PATHS[9]="/workspace/RLinf/logs/20260123-11:48:31-libero_object_task9_lora_residual_sac_openvlaoft.yaml/libero_object_task9_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model"

# 要跳过的task IDs（不进行分析）
SKIP_TASKS=(0 3 4 5)

# 状态收集配置
STATE_METHOD="base_rollout"                      # 状态收集方法: demo, both_demo, base_rollout
MAX_DEMOS=5                                   # 用于状态收集的episode/demo数量
REFERENCE_TASK="0"                            # 参考任务ID（仅当state_method=demo时使用）

# 分析阈值配置
DELTA_DIR=0.5                                 # 方向一致性危险阈值（0-1之间）

# 输出目录（可选，用于汇总报告）
OUTPUT_BASE_DIR="/workspace/RLinf/results"  # 如果为空，将使用config中的默认输出目录

# =============================================================================
# 脚本执行区域 - 通常不需要修改
# =============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "Batch Correction Vector Field Analysis (rlinf LoRA Residual Policy)"
echo "=============================================================================="
echo ""

# 获取所有要分析的task IDs（排除跳过的task）
TASK_IDS=()
for task_id in "${!CHECKPOINT_PATHS[@]}"; do
    # 检查是否在跳过列表中
    skip=false
    for skip_task in "${SKIP_TASKS[@]}"; do
        if [ "$task_id" == "$skip_task" ]; then
            skip=true
            break
        fi
    done
    if [ "$skip" == false ]; then
        TASK_IDS+=("$task_id")
    fi
done

# 排序task IDs
TASK_IDS=($(printf '%s\n' "${TASK_IDS[@]}" | sort -n))

echo "Configuration:"
echo "  Config File:      $CONFIG_FILE"
echo "  State Method:     $STATE_METHOD"
echo "  Max Demos:        $MAX_DEMOS"
echo "  Delta Dir:        $DELTA_DIR"
echo "  Skip Tasks:       ${SKIP_TASKS[*]}"
echo "  Tasks to analyze: ${TASK_IDS[*]}"
echo "  Total tasks:      ${#TASK_IDS[@]}"
echo ""

# 验证checkpoint路径
echo "Checkpoint paths:"
for task_id in "${TASK_IDS[@]}"; do
    ckpt_path="${CHECKPOINT_PATHS[$task_id]}"
    if [ -n "$ckpt_path" ]; then
        if [ ! -d "$ckpt_path" ] && [ ! -f "$ckpt_path" ]; then
            echo "  ⚠️  Task $task_id: $ckpt_path (path not found, will use config default)"
        else
            echo "  ✅ Task $task_id: $ckpt_path"
        fi
    else
        echo "  ℹ️  Task $task_id: (using config default)"
    fi
done
echo ""

# 计算总对数
TOTAL_PAIRS=$(( ${#TASK_IDS[@]} * (${#TASK_IDS[@]} - 1) / 2 ))
CURRENT_PAIR=0

echo "=============================================================================="
echo "Starting batch analysis for ${#TASK_IDS[@]} tasks (${TOTAL_PAIRS} pairs)"
echo "=============================================================================="
echo ""

# 初始化结果数组
declare -a RESULTS

# 遍历所有task pairs
for i in "${!TASK_IDS[@]}"; do
    task_i="${TASK_IDS[$i]}"
    ckpt_i="${CHECKPOINT_PATHS[$task_i]}"
    
    for j in "${!TASK_IDS[@]}"; do
        task_j="${TASK_IDS[$j]}"
        ckpt_j="${CHECKPOINT_PATHS[$task_j]}"
        
        # 只分析 i < j 的pairs（避免重复）
        if [ "$task_i" -ge "$task_j" ]; then
            continue
        fi
        
        CURRENT_PAIR=$((CURRENT_PAIR + 1))
        
        echo "=============================================================================="
        echo "[$CURRENT_PAIR/$TOTAL_PAIRS] Analyzing Task $task_i vs Task $task_j"
        echo "=============================================================================="
        echo "Checkpoint I: ${ckpt_i:-'(using config default)'}"
        echo "Checkpoint J: ${ckpt_j:-'(using config default)'}"
        echo ""
        
        # 构建Python命令
        PYTHON_CMD="python analyze_correction_vector_field.py"
        PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
        PYTHON_CMD="$PYTHON_CMD --task_i $task_i"
        PYTHON_CMD="$PYTHON_CMD --task_j $task_j"
        PYTHON_CMD="$PYTHON_CMD --state_method $STATE_METHOD"
        PYTHON_CMD="$PYTHON_CMD --max_demos $MAX_DEMOS"
        PYTHON_CMD="$PYTHON_CMD --delta_dir $DELTA_DIR"
        
        # 添加可选的checkpoint路径
        if [ -n "$ckpt_i" ]; then
            PYTHON_CMD="$PYTHON_CMD --checkpoint_i $ckpt_i"
        fi
        
        if [ -n "$ckpt_j" ]; then
            PYTHON_CMD="$PYTHON_CMD --checkpoint_j $ckpt_j"
        fi
        
        # 添加可选的reference_task（仅当state_method=demo时使用）
        if [ -n "$REFERENCE_TASK" ] && [ "$STATE_METHOD" == "demo" ]; then
            PYTHON_CMD="$PYTHON_CMD --reference_task $REFERENCE_TASK"
        fi
        
        # 执行分析
        echo "Running: $PYTHON_CMD"
        echo ""
        
        if eval $PYTHON_CMD; then
            echo "✅ Task $task_i vs Task $task_j: Analysis completed"
            RESULTS+=("Task_${task_i}_vs_Task_${task_j}: SUCCESS")
        else
            echo "❌ Task $task_i vs Task $task_j: Analysis failed"
            RESULTS+=("Task_${task_i}_vs_Task_${task_j}: FAILED")
        fi
        
        echo ""
        echo "---"
        echo ""
    done
done

# 生成汇总报告
echo "=============================================================================="
echo "Generating Summary Report"
echo "=============================================================================="
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "$OUTPUT_BASE_DIR" ]; then
    SUMMARY_FILE="${OUTPUT_BASE_DIR}/batch_correction_field_analysis_${TIMESTAMP}/summary_all_pairs.txt"
    mkdir -p "$(dirname "$SUMMARY_FILE")"
else
    SUMMARY_FILE="${SCRIPT_DIR}/batch_correction_field_analysis_${TIMESTAMP}_summary.txt"
fi

{
    echo "Batch Correction Vector Field Analysis Summary"
    echo "=============================================="
    echo "Timestamp: $(date)"
    echo "Config File: $CONFIG_FILE"
    echo "State Method: $STATE_METHOD"
    echo "Max Demos: $MAX_DEMOS"
    echo "Delta Dir: $DELTA_DIR"
    echo "Skip Tasks: ${SKIP_TASKS[*]}"
    echo ""
    echo "Total Tasks Analyzed: ${#TASK_IDS[@]}"
    echo "Tasks: ${TASK_IDS[*]}"
    echo "Total Pairs Analyzed: $TOTAL_PAIRS"
    echo ""
    echo "Checkpoint Paths:"
    for task_id in "${TASK_IDS[@]}"; do
        ckpt_path="${CHECKPOINT_PATHS[$task_id]}"
        echo "  Task $task_id: ${ckpt_path:-'(using config default)'}"
    done
    echo ""
    echo "Results:"
    for result in "${RESULTS[@]}"; do
        echo "  $result"
    done
    echo ""
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "=============================================================================="
echo "✅ Batch analysis completed!"
echo "=============================================================================="
echo ""
echo "Results summary:"
cat "$SUMMARY_FILE"

