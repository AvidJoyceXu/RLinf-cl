#!/bin/bash

################################################################################
# Evaluate Merged Policies Script
# 
# 功能说明：
#   本脚本用于评估merged_policy_dir下合并后的policy。
#   对于每个task(i,j)组合的合并policy，分别在task i和task j上进行评估。
#
# 使用方法：
#   bash eval_merged_policies.sh [config_name] [merged_policy_dir]
#
# 参数说明：
#   config_name: Hydra配置文件名（不含.yaml后缀），默认为 libero_spatial_task0_lora_residual_sac_openvlaoft
#   merged_policy_dir: 合并后的policy目录，默认为 merged_policy_dir
#
# 示例：
#   bash eval_merged_policies.sh libero_spatial_task0_lora_residual_sac_openvlaoft merged_policy_dir
#
# 工作原理：
#   1. 扫描merged_policy_dir下的所有task(i,j)目录
#   2. 对于每个合并后的policy，分别在task i和task j上进行评估
#   3. 使用env.eval.specific_reset_id来指定评估的task ID
#   4. 每个测试的结果会保存到独立的日志目录中
#   5. 最后生成一个汇总文件，包含所有测试的结果
#
# 输出：
#   - 结果保存在: ${REPO_PATH}/results/merged_policy_eval_YYYYMMDD-HH:MM:SS/
#   - 每个测试的日志: task{i}_{j}/task{i}_{j}_eval_task{k}/eval.log
#   - 汇总文件: evaluation_summary.txt
################################################################################

# 设置错误处理：遇到错误不立即退出，继续执行后续测试
set +e

# ============================================================================
# 环境变量设置
# ============================================================================

# 获取脚本所在目录的绝对路径（仓库根目录）
export REPO_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取embodiment目录路径
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
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
# 参数解析和验证
# ============================================================================

# 默认配置名称
BASE_CONFIG_NAME=${1:-"libero_spatial_task0_lora_residual_sac_openvlaoft"}

# 默认合并policy目录
MERGED_POLICY_DIR=${2:-"merged_policy_dir"}

# 检查merged_policy_dir是否存在
if [ ! -d "${MERGED_POLICY_DIR}" ]; then
    echo "错误: 合并policy目录不存在: ${MERGED_POLICY_DIR}"
    echo ""
    echo "使用方法:"
    echo "  bash eval_merged_policies.sh [config_name] [merged_policy_dir]"
    echo ""
    echo "示例:"
    echo "  bash eval_merged_policies.sh libero_spatial_task0_lora_residual_sac_openvlaoft merged_policy_dir"
    exit 1
fi

# ============================================================================
# 扫描合并后的policy目录
# ============================================================================

echo "扫描合并后的policy目录: ${MERGED_POLICY_DIR}"
echo ""

# 存储所有找到的task组合
declare -a TASK_PAIRS=()

# 扫描merged_policy_dir下的所有task(i,j)目录
# 格式: task{i}_{j} 或 task{i}_{j}/
for dir in "${MERGED_POLICY_DIR}"/task*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # 提取目录名（去掉路径前缀）
    dirname=$(basename "$dir")
    
    # 匹配格式: task{i}_{j}
    if [[ "$dirname" =~ ^task([0-9]+)_([0-9]+)$ ]]; then
        task_i="${BASH_REMATCH[1]}"
        task_j="${BASH_REMATCH[2]}"
        TASK_PAIRS+=("${task_i}_${task_j}")
        echo "✓ 找到合并policy: ${dirname} (Task ${task_i} + Task ${task_j})"
    fi
done

if [ ${#TASK_PAIRS[@]} -eq 0 ]; then
    echo "错误: 在 ${MERGED_POLICY_DIR} 下没有找到任何task(i,j)格式的目录"
    echo "      期望格式: task{i}_{j}/ (例如: task0_2/)"
    exit 1
fi

echo ""
echo "共找到 ${#TASK_PAIRS[@]} 个合并后的policy"
echo ""

# ============================================================================
# 结果目录设置
# ============================================================================

# 创建带时间戳的结果目录
RESULTS_DIR="${REPO_PATH}/results/merged_policy_eval_$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${RESULTS_DIR}"

# 汇总文件路径
SUMMARY_FILE="${RESULTS_DIR}/evaluation_summary.txt"

# 初始化汇总文件
{
    echo "=========================================="
    echo "Merged Policy Evaluation Summary"
    echo "=========================================="
    echo "Base Config: ${BASE_CONFIG_NAME}"
    echo "Merged Policy Dir: ${MERGED_POLICY_DIR}"
    echo "Start Time: $(date)"
    echo ""
    echo "Found ${#TASK_PAIRS[@]} merged policies:"
    for pair in "${TASK_PAIRS[@]}"; do
        echo "  Task Pair: ${pair}"
    done
    echo ""
    echo "=========================================="
    echo "Test Results (Format: Merged_Task{i}_{j} -> Eval_Task{k}: Result)"
    echo "=========================================="
} > "${SUMMARY_FILE}"

# ============================================================================
# 测试执行
# ============================================================================

# 计数器
total_tests=0
completed_tests=0
failed_tests=0

# 遍历所有合并后的policy
for pair in "${TASK_PAIRS[@]}"; do
    # 解析task i和task j
    task_i=$(echo "$pair" | cut -d'_' -f1)
    task_j=$(echo "$pair" | cut -d'_' -f2)
    
    # 合并后的policy路径
    MERGED_POLICY_PATH="${MERGED_POLICY_DIR}/task${task_i}_${task_j}"
    
    # 检查policy路径是否存在
    if [ ! -d "$MERGED_POLICY_PATH" ]; then
        echo "警告: 合并policy路径不存在: ${MERGED_POLICY_PATH}"
        continue
    fi
    
    # 对每个合并后的policy，分别在task i和task j上进行评估
    for eval_task_id in "$task_i" "$task_j"; do
        total_tests=$((total_tests + 1))
        
        echo ""
        echo "=========================================="
        echo "测试 ${total_tests}"
        echo "合并Policy: Task ${task_i} + Task ${task_j}"
        echo "评估任务: Task ${eval_task_id}"
        echo "=========================================="
        
        # 为当前测试创建独立的日志目录
        # 格式: task{i}_{j}/task{i}_{j}_eval_task{k}/
        TEST_LOG_DIR="${RESULTS_DIR}/task${task_i}_${task_j}/task${task_i}_${task_j}_eval_task${eval_task_id}"
        mkdir -p "${TEST_LOG_DIR}"
        
        # ====================================================================
        # 构建Hydra命令
        # ====================================================================
        # 使用Hydra override来修改配置：
        # 1. runner.eval_policy_path: 设置为合并后的policy路径
        # 2. env.eval.specific_reset_id: 设置为当前评估的task ID
        # 3. runner.only_eval: 设置为True，只进行评估不训练
        # 4. runner.logger.log_path: 设置为当前测试的日志目录
        CMD="python ${SRC_FILE} \
            --config-path ${EMBODIED_PATH}/config/ \
            --config-name ${BASE_CONFIG_NAME} \
            runner.eval_policy_path=\"${MERGED_POLICY_PATH}\" \
            env.eval.specific_reset_id=${eval_task_id} \
            runner.only_eval=True \
            runner.logger.log_path=${TEST_LOG_DIR}"
        
        echo "执行命令:"
        echo "  ${CMD}"
        echo "合并Policy路径: ${MERGED_POLICY_PATH}"
        echo "评估Task ID: ${eval_task_id}"
        
        # ====================================================================
        # 执行评估
        # ====================================================================
        LOG_FILE="${TEST_LOG_DIR}/eval.log"
        
        # 运行评估命令，将输出保存到日志文件
        if ${CMD} > "${LOG_FILE}" 2>&1; then
            completed_tests=$((completed_tests + 1))
            echo "✓ 成功完成: Merged_Task${task_i}_${task_j} -> Eval_Task_${eval_task_id}"
            
            # 尝试从日志文件中提取成功率（根据实际日志格式调整）
            # 这里使用grep查找包含"success"的行，提取数字
            SUCCESS_RATE=$(grep -iE "success.*rate|success_rate|success rate|success.*:" "${LOG_FILE}" | tail -n 1 | grep -oE "[0-9]+\.[0-9]+|[0-9]+%" | head -n 1 || echo "N/A")
            
            # 记录到汇总文件
            echo "Merged_Task${task_i}_${task_j} -> Eval_Task_${eval_task_id}: SUCCESS (Rate: ${SUCCESS_RATE})" >> "${SUMMARY_FILE}"
        else
            failed_tests=$((failed_tests + 1))
            echo "✗ 失败: Merged_Task${task_i}_${task_j} -> Eval_Task_${eval_task_id}"
            echo "Merged_Task${task_i}_${task_j} -> Eval_Task_${eval_task_id}: FAILED" >> "${SUMMARY_FILE}"
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
    echo "Total Tests: ${total_tests}"
    echo "Completed: ${completed_tests}"
    echo "Failed: ${failed_tests}"
    if [ ${total_tests} -gt 0 ]; then
        SUCCESS_PERCENT=$(echo "scale=2; ${completed_tests} * 100 / ${total_tests}" | bc)
        echo "Success Rate: ${SUCCESS_PERCENT}%"
    fi
    echo "=========================================="
} >> "${SUMMARY_FILE}"

# 打印最终汇总到控制台
echo ""
echo "=========================================="
echo "合并Policy评估完成！"
echo "=========================================="
echo "总测试数: ${total_tests}"
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
