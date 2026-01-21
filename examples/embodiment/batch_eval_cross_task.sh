#!/bin/bash

################################################################################
# Batch Cross-Task Evaluation Script
# 
# 功能说明：
#   本脚本用于在多个task上训练的policy之间进行交叉任务评估。
#   注意：会跳过task 1, 8, 9的policy和测试。
#   总共进行49次测试（7个policy × 7个task = 49）。
#   有效的task ID: 0, 2, 3, 4, 5, 6, 7
#
# 使用方法：
#   bash batch_eval_cross_task.sh [config_name] [policy_path_0] [policy_path_1] ... [policy_path_9]
#
# 参数说明：
#   config_name: Hydra配置文件名（不含.yaml后缀），默认为 libero_spatial_task0_lora_residual_sac_gr00t
#   policy_path_0 到 policy_path_9: 10个policy的路径，分别对应task 0-9上训练的policy
#   注意：虽然需要提供10个路径，但task 1, 8, 9的policy和测试会被跳过
#
# 示例：
#   bash examples/embodiment/batch_eval_cross_task.sh libero_spatial_task0_lora_residual_sac_gr00t \
#       /workspace/RLinf/logs/20260121-07:24:09-libero_spatial_task0_lora_residual_sac_openvlaoft/task0_single_trial_lora_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#       /skip \    
#       /workspace/RLinf/logs/20260119-10:05:34-libero_spatial_task2_lora_residual_sac_openvlaoft/task2_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#       /workspace/RLinf/logs/20260119-13:18:01-libero_spatial_task3_lora_residual_sac_openvlaoft/task3_single_trial_residual_sac_openvlaoft/checkpoints/global_step_3000/actor/huggingface_model \
#       /workspace/RLinf/logs/20260119-16:57:44-libero_spatial_task4_lora_residual_sac_openvlaoft/task4_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#       /skip \
#       /workspace/RLinf/logs/20260120-02:34:04-libero_spatial_task6_lora_residual_sac_openvlaoft/task6_single_trial_residual_sac_openvlaoft/checkpoints/global_step_4000/actor/huggingface_model \
#       /workspace/RLinf/logs/20260120-02:34:18-libero_spatial_task7_lora_residual_sac_openvlaoft/task7_single_trial_residual_sac_openvlaoft/checkpoints/global_step_4000/actor/huggingface_model \
#       /skip \    
#       /skip 

# 工作原理：
#   1. 脚本会遍历所有policy-task组合，但跳过task 1, 8, 9（7×7=49种）
#   2. 对于每个组合，使用Hydra override来设置：
#      - runner.eval_policy_path: 当前policy的路径
#      - env.eval.specific_reset_id: 当前测试的task ID (0, 2-7)
#   3. 每个测试的结果会保存到独立的日志目录中
#   4. 最后生成一个汇总文件，包含所有测试的结果
#
# 输出：
#   - 结果保存在: ${REPO_PATH}/results/batch_eval_cross_task_YYYYMMDD-HH:MM:SS/
#   - 每个测试的日志: policy_task{P}_eval_task{E}/eval.log
#   - 汇总文件: evaluation_summary.txt
################################################################################

# 设置错误处理：遇到错误不立即退出，继续执行后续测试
set +e

# ============================================================================
# 环境变量设置
# ============================================================================

# 获取脚本所在目录的绝对路径
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取仓库根目录（embodiment目录的父目录的父目录）
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
# 评估脚本路径
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

# MuJoCo渲染设置（使用osmesa进行无头渲染）
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# CUDA和Hydra设置
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# ============================================================================
# 跳过任务配置
# ============================================================================
# 定义要跳过的task ID列表（task 1, 8, 9会被跳过）
# 有效的task ID: 0, 2, 3, 4, 5, 6, 7
declare -a SKIP_TASKS=(1 8 9)

# 检查task是否应该被跳过
should_skip_task() {
    local task_id=$1
    for skip_task in "${SKIP_TASKS[@]}"; do
        if [ "$task_id" -eq "$skip_task" ]; then
            return 0  # 应该跳过
        fi
    done
    return 1  # 不应该跳过
}

# ============================================================================
# 参数解析和验证
# ============================================================================

# 默认配置名称
BASE_CONFIG_NAME=${1:-"libero_spatial_task0_lora_residual_sac_gr00t"}

# 检查是否提供了足够的policy路径参数
# 需要至少1个参数（config_name），然后需要10个policy路径
# 注意：即使某些task会被跳过，仍需要提供完整的10个路径参数
if [ $# -lt 11 ]; then
    echo "错误: 需要提供10个policy路径参数"
    echo ""
    echo "实际接收到的参数数量: $#"
    echo "参数列表:"
    for i in $(seq 1 $#); do
        echo "  [$i] ${!i}"
    done
    echo ""
    echo "使用方法:"
    echo "  bash batch_eval_cross_task.sh [config_name] [policy_path_0] [policy_path_1] ... [policy_path_9]"
    echo ""
    echo "注意:"
    echo "  - policy_path可以是目录路径（推荐）或文件路径"
    echo "  - 即使task 1, 8, 9会被跳过，仍需要提供完整的10个路径参数"
    echo "  - 使用多行命令时，确保每行末尾的反斜杠后没有注释"
    echo ""
    echo "示例:"
    echo "  bash batch_eval_cross_task.sh libero_spatial_task0_lora_residual_sac_gr00t \\"
    echo "      /path/to/policy_task0 \\"
    echo "      /path/to/policy_task1 \\"
    echo "      ..."
    echo "      /path/to/policy_task9"
    exit 1
fi

# ============================================================================
# Policy路径列表（全局变量）
# ============================================================================
# 存储10个task上训练的policy路径
# 索引0-9分别对应task 0-9上训练的policy
declare -a POLICY_PATHS=(
    "${2}"   # policy_path_0: task 0上训练的policy
    "${3}"   # policy_path_1: task 1上训练的policy
    "${4}"   # policy_path_2: task 2上训练的policy
    "${5}"   # policy_path_3: task 3上训练的policy
    "${6}"   # policy_path_4: task 4上训练的policy
    "${7}"   # policy_path_5: task 5上训练的policy
    "${8}"   # policy_path_6: task 6上训练的policy
    "${9}"   # policy_path_7: task 7上训练的policy
    "${10}"  # policy_path_8: task 8上训练的policy
    "${11}"  # policy_path_9: task 9上训练的policy
)

# 验证所有policy路径是否存在（跳过task 1, 8, 9）
echo "验证policy路径（跳过task 1, 8, 9）..."
echo "注意: runner.eval_policy_path 接受目录路径（推荐）或文件路径"
echo ""
for i in {0..9}; do
    if should_skip_task $i; then
        echo "⊘ 跳过验证 (Task $i): 该task会被跳过"
        continue
    fi
    
    POLICY_PATH="${POLICY_PATHS[$i]}"
    # 检查路径是否存在（可以是目录或文件）
    if [ -d "$POLICY_PATH" ]; then
        echo "✓ Policy路径有效 (Task $i): ${POLICY_PATH} [目录]"
    elif [ -f "$POLICY_PATH" ]; then
        echo "✓ Policy路径有效 (Task $i): ${POLICY_PATH} [文件]"
    else
        echo "警告: Policy路径不存在 (Task $i): ${POLICY_PATH}"
        echo "      该路径的测试可能会失败"
    fi
done
echo ""

# ============================================================================
# 结果目录设置
# ============================================================================

# 创建带时间戳的结果目录
RESULTS_DIR="${REPO_PATH}/results/batch_eval_cross_task_$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${RESULTS_DIR}"

# 汇总文件路径
SUMMARY_FILE="${RESULTS_DIR}/evaluation_summary.txt"

# 初始化汇总文件
{
    echo "=========================================="
    echo "Batch Cross-Task Evaluation Summary"
    echo "=========================================="
    echo "Base Config: ${BASE_CONFIG_NAME}"
    echo "Start Time: $(date)"
    echo ""
    echo "Skipped Tasks: ${SKIP_TASKS[@]}"
    echo "Valid Tasks: 0, 2, 3, 4, 5, 6, 7"
    echo "Total Tests: 7 × 7 = 49"
    echo ""
    echo "Policy Paths:"
    for i in {0..9}; do
        if should_skip_task $i; then
            echo "  Task $i: ${POLICY_PATHS[$i]} (SKIPPED)"
        else
            echo "  Task $i: ${POLICY_PATHS[$i]}"
        fi
    done
    echo ""
    echo "=========================================="
    echo "Test Results (Format: Policy_Task_ID -> Eval_Task_ID: Result)"
    echo "=========================================="
} > "${SUMMARY_FILE}"

# ============================================================================
# 测试执行
# ============================================================================

# 计数器
total_tests=0
completed_tests=0
failed_tests=0
skipped_tests=0

# 遍历所有policy-task组合（跳过task 1, 8, 9）
# policy_task_id: 0-9，表示使用哪个task上训练的policy（跳过1, 8, 9）
# eval_task_id: 0-9，表示在哪个task上进行评估（跳过1, 8, 9）
for policy_task_id in {0..9}; do
    # 跳过policy来源task 1, 8, 9
    if should_skip_task $policy_task_id; then
        continue
    fi
    
    for eval_task_id in {0..9}; do
        # 跳过评估task 1, 8, 9
        if should_skip_task $eval_task_id; then
            skipped_tests=$((skipped_tests + 1))
            continue
        fi
        
        total_tests=$((total_tests + 1))
        
        echo ""
        echo "=========================================="
        echo "测试 ${total_tests}/49"
        echo "Policy来源: Task ${policy_task_id}"
        echo "评估任务: Task ${eval_task_id}"
        echo "=========================================="
        
        # 获取当前policy路径
        POLICY_PATH="${POLICY_PATHS[$policy_task_id]}"
        
        # 再次检查policy路径是否存在（可以是目录或文件）
        if [ ! -d "$POLICY_PATH" ] && [ ! -f "$POLICY_PATH" ]; then
            echo "错误: Policy路径不存在: ${POLICY_PATH}"
            failed_tests=$((failed_tests + 1))
            echo "Policy_Task_${policy_task_id} -> Eval_Task_${eval_task_id}: SKIPPED (path not found: ${POLICY_PATH})" >> "${SUMMARY_FILE}"
            continue
        fi
        
        # 为当前测试创建独立的日志目录
        TEST_LOG_DIR="${RESULTS_DIR}/policy_task${policy_task_id}_eval_task${eval_task_id}"
        mkdir -p "${TEST_LOG_DIR}"
        
        # ====================================================================
        # 构建Hydra命令
        # ====================================================================
        # 使用Hydra override来修改配置：
        # 1. runner.eval_policy_path: 设置为当前policy的路径
        # 2. env.eval.specific_reset_id: 设置为当前评估的task ID
        # 3. runner.only_eval: 设置为True，只进行评估不训练
        # 4. runner.logger.log_path: 设置为当前测试的日志目录
        CMD="python ${SRC_FILE} \
            --config-path ${EMBODIED_PATH}/config/ \
            --config-name ${BASE_CONFIG_NAME} \
            runner.eval_policy_path=\"${POLICY_PATH}\" \
            env.eval.specific_reset_id=${eval_task_id} \
            runner.only_eval=True \
            runner.logger.log_path=${TEST_LOG_DIR}"
        
        echo "执行命令:"
        echo "  ${CMD}"
        echo "Policy路径: ${POLICY_PATH}"
        echo "评估Task ID: ${eval_task_id}"
        
        # ====================================================================
        # 执行评估
        # ====================================================================
        LOG_FILE="${TEST_LOG_DIR}/eval.log"
        
        # 运行评估命令，将输出保存到日志文件
        if ${CMD} > "${LOG_FILE}" 2>&1; then
            completed_tests=$((completed_tests + 1))
            echo "✓ 成功完成: Policy_Task_${policy_task_id} -> Eval_Task_${eval_task_id}"
            
            # 尝试从日志文件中提取成功率（根据实际日志格式调整）
            # 这里使用grep查找包含"success"的行，提取数字
            SUCCESS_RATE=$(grep -iE "success.*rate|success_rate|success rate|success.*:" "${LOG_FILE}" | tail -n 1 | grep -oE "[0-9]+\.[0-9]+|[0-9]+%" | head -n 1 || echo "N/A")
            
            # 记录到汇总文件
            echo "Policy_Task_${policy_task_id} -> Eval_Task_${eval_task_id}: SUCCESS (Rate: ${SUCCESS_RATE})" >> "${SUMMARY_FILE}"
        else
            failed_tests=$((failed_tests + 1))
            echo "✗ 失败: Policy_Task_${policy_task_id} -> Eval_Task_${eval_task_id}"
            echo "Policy_Task_${policy_task_id} -> Eval_Task_${eval_task_id}: FAILED" >> "${SUMMARY_FILE}"
        fi
        
        echo "日志文件: ${LOG_FILE}"
    done
done

# ============================================================================
# 生成最终汇总
# ============================================================================

{
    echo ""
    echo "=========================================="
    echo "Evaluation Complete"
    echo "=========================================="
    echo "End Time: $(date)"
    echo "Skipped Tasks: ${SKIP_TASKS[@]}"
    echo "Total Tests: ${total_tests} (7 × 7 = 49)"
    echo "Completed: ${completed_tests}"
    echo "Failed: ${failed_tests}"
    echo "Success Rate: $(echo "scale=2; ${completed_tests} * 100 / ${total_tests}" | bc)%"
    echo "=========================================="
} >> "${SUMMARY_FILE}"

# 打印最终汇总到控制台
echo ""
echo "=========================================="
echo "批量交叉任务评估完成！"
echo "=========================================="
echo "跳过的任务: ${SKIP_TASKS[@]}"
echo "有效任务: 0, 2, 3, 4, 5, 6, 7"
echo "总测试数: ${total_tests} (7 × 7 = 49)"
echo "成功完成: ${completed_tests}"
echo "失败: ${failed_tests}"
if [ ${total_tests} -gt 0 ]; then
    SUCCESS_PERCENT=$(echo "scale=2; ${completed_tests} * 100 / ${total_tests}" | bc)
    echo "成功率: ${SUCCESS_PERCENT}%"
fi
echo ""
echo "结果保存目录: ${RESULTS_DIR}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="

# 退出码：如果有失败的测试，返回非零退出码
if [ ${failed_tests} -gt 0 ]; then
    exit 1
else
    exit 0
fi

