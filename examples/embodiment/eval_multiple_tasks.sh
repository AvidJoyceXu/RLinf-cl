#!/bin/bash

################################################################################
# Multiple Tasks Evaluation Script (批量单任务测试脚本)
# 
# 功能说明：
#   本脚本用于指定单一policy path和多个specific task id，进行批量单任务测试。
#   适用于需要对多个任务进行评估的场景。
#
# 使用方法：
#   bash eval_multiple_tasks.sh [config_name] [policy_path] [task_id1] [task_id2] ... [task_idN] [--seed SEED]
#
# 参数说明：
#   config_name: Hydra配置文件名（不含.yaml后缀），默认为 libero_spatial_task0_lora_residual_sac_gr00t
#   policy_path: 要评估的policy路径（可以是目录路径或文件路径）
#   task_id1, task_id2, ...: 要评估的task ID列表 (0-9)
#   --seed SEED: 可选，设置环境随机种子 (env.seed)，默认为配置文件中的值
#
# 示例：
#   bash examples/embodiment/eval_multiple_tasks.sh libero_spatial_task0_lora_residual_sac_gr00t \
#       /workspace/RLinf/logs/20260121-07:24:09-libero_spatial_task0_lora_residual_sac_openvlaoft/task0_single_trial_lora_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
#       0 1 2 3 --seed 42
#
# 工作原理：
#   1. 对每个task_id循环执行评估
#   2. 使用Hydra override来设置：
#      - runner.eval_policy_path: 指定的policy路径
#      - env.eval.specific_reset_id: 当前循环的task ID
#      - runner.only_eval: 设置为True，只进行评估不训练
#      - env.seed: 如果提供了--seed参数，则设置为指定值
#   3. 每个task的测试结果会保存到独立的日志目录中
#   4. 最后输出所有任务的评估结果汇总
#
# 输出：
#   - 每个task的结果保存在: ${REPO_PATH}/results/multi_eval_YYYYMMDD-HH:MM:SS/task_${TASK_ID}/
#   - 每个task的日志文件: eval.log
#   - 汇总信息: evaluation_summary.txt
################################################################################

# 设置错误处理：遇到错误立即退出
set -e

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
# 参数解析和验证
# ============================================================================

# 初始化变量
BASE_CONFIG_NAME=""
POLICY_PATH=""
TASK_IDS=()
ENV_SEED=""

# 解析参数（支持 --seed 选项）
# 先收集位置参数，然后处理选项参数
POSITIONAL_ARGS=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    
    if [ "$arg" = "--seed" ]; then
        # 检查是否有下一个参数
        if [ $i -ge $# ]; then
            echo "错误: --seed 选项需要一个值"
            exit 1
        fi
        i=$((i + 1))
        ENV_SEED="${!i}"
        
        # 验证seed是否为有效数字
        if ! [[ "$ENV_SEED" =~ ^[0-9]+$ ]]; then
            echo "错误: seed 必须是数字，当前值: ${ENV_SEED}"
            exit 1
        fi
    else
        # 收集位置参数
        POSITIONAL_ARGS+=("$arg")
    fi
    
    i=$((i + 1))
done

# 处理位置参数
if [ ${#POSITIONAL_ARGS[@]} -lt 2 ]; then
    echo "错误: 需要提供至少2个位置参数 (config_name 和 policy_path)"
    exit 1
fi

# 第一个位置参数是config_name
BASE_CONFIG_NAME="${POSITIONAL_ARGS[0]}"

# 第二个位置参数是policy_path
POLICY_PATH="${POSITIONAL_ARGS[1]}"

# 剩余的位置参数是task_ids
for i in $(seq 2 $((${#POSITIONAL_ARGS[@]} - 1))); do
    TASK_ID="${POSITIONAL_ARGS[$i]}"
    
    # 验证task_id是否为有效数字
    if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
        echo "错误: task_id 必须是数字，当前值: ${TASK_ID}"
        exit 1
    fi
    
    # 验证task_id范围
    if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -gt 9 ]; then
        echo "错误: task_id 必须在 0-9 之间，当前值: ${TASK_ID}"
        exit 1
    fi
    
    TASK_IDS+=("${TASK_ID}")
done

# 检查必需参数
if [ -z "$POLICY_PATH" ]; then
    echo "错误: 需要提供policy_path参数"
    echo ""
    echo "使用方法:"
    echo "  bash eval_multiple_tasks.sh [config_name] [policy_path] [task_id1] [task_id2] ... [task_idN] [--seed SEED]"
    echo ""
    echo "参数说明:"
    echo "  config_name: Hydra配置文件名（不含.yaml后缀），可选，默认为 libero_spatial_task0_lora_residual_sac_gr00t"
    echo "  policy_path: 要评估的policy路径（可以是目录路径或文件路径），必需"
    echo "  task_id1, task_id2, ...: 要评估的task ID列表 (0-9)，至少需要一个，必需"
    echo "  --seed SEED: 可选，设置环境随机种子 (env.seed)"
    echo ""
    echo "示例:"
    echo "  bash examples/embodiment/eval_multiple_tasks.sh libero_spatial_task0_lora_residual_sac_gr00t \\"
    echo "      /path/to/policy \\"
    echo "      0 1 2 3 --seed 42"
    echo ""
    echo "注意:"
    echo "  - policy_path可以是目录路径（推荐）或文件路径"
    echo "  - task_id 应该是 0-9 之间的整数"
    exit 1
fi

if [ ${#TASK_IDS[@]} -eq 0 ]; then
    echo "错误: 需要提供至少一个task_id"
    echo ""
    echo "使用方法:"
    echo "  bash eval_multiple_tasks.sh [config_name] [policy_path] [task_id1] [task_id2] ... [task_idN] [--seed SEED]"
    echo ""
    echo "参数说明:"
    echo "  config_name: Hydra配置文件名（不含.yaml后缀），可选，默认为 libero_spatial_task0_lora_residual_sac_gr00t"
    echo "  policy_path: 要评估的policy路径（可以是目录路径或文件路径），必需"
    echo "  task_id1, task_id2, ...: 要评估的task ID列表 (0-9)，至少需要一个，必需"
    echo "  --seed SEED: 可选，设置环境随机种子 (env.seed)"
    exit 1
fi

# 验证policy路径是否存在（可以是目录或文件）
if [ ! -d "$POLICY_PATH" ] && [ ! -f "$POLICY_PATH" ]; then
    echo "错误: Policy路径不存在: ${POLICY_PATH}"
    echo "      请检查路径是否正确"
    exit 1
fi

# 显示policy路径类型
if [ -d "$POLICY_PATH" ]; then
    echo "✓ Policy路径有效: ${POLICY_PATH} [目录]"
else
    echo "✓ Policy路径有效: ${POLICY_PATH} [文件]"
fi

# 显示要测试的task列表
echo "✓ 待测试的Task ID列表: ${TASK_IDS[@]}"

# 显示seed设置
if [ -n "$ENV_SEED" ]; then
    echo "✓ 环境随机种子: ${ENV_SEED}"
else
    echo "✓ 环境随机种子: 使用配置文件默认值"
fi
echo ""

# ============================================================================
# 结果目录设置
# ============================================================================

# 创建带时间戳的结果目录
RESULTS_BASE_DIR="${REPO_PATH}/results/multi_eval_$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${RESULTS_BASE_DIR}"

# 汇总文件路径
SUMMARY_FILE="${RESULTS_BASE_DIR}/evaluation_summary.txt"

# 初始化汇总文件
{
    echo "=========================================="
    echo "Multiple Tasks Evaluation Summary"
    echo "=========================================="
    echo "Base Config: ${BASE_CONFIG_NAME}"
    echo "Policy Path: ${POLICY_PATH}"
    echo "Task IDs: ${TASK_IDS[@]}"
    echo "Total Tasks: ${#TASK_IDS[@]}"
    if [ -n "$ENV_SEED" ]; then
        echo "Environment Seed: ${ENV_SEED}"
    fi
    echo "Start Time: $(date)"
    echo ""
} > "${SUMMARY_FILE}"

# ============================================================================
# 测试执行
# ============================================================================

echo ""
echo "=========================================="
echo "开始批量评估"
echo "=========================================="
echo "配置名称: ${BASE_CONFIG_NAME}"
echo "Policy路径: ${POLICY_PATH}"
echo "评估Task ID列表: ${TASK_IDS[@]}"
if [ -n "$ENV_SEED" ]; then
    echo "环境随机种子: ${ENV_SEED}"
fi
echo "结果目录: ${RESULTS_BASE_DIR}"
echo "=========================================="
echo ""

# 记录成功和失败的任务
SUCCESSFUL_TASKS=()
FAILED_TASKS=()
TASK_RESULTS=()

# 循环处理每个task_id
for TASK_ID in "${TASK_IDS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "正在评估 Task ID: ${TASK_ID}"
    echo "----------------------------------------"
    
    # 为每个task创建独立的结果目录
    TASK_RESULTS_DIR="${RESULTS_BASE_DIR}/task_${TASK_ID}"
    mkdir -p "${TASK_RESULTS_DIR}"
    
    # ====================================================================
    # 构建Hydra命令（与原始脚本完全一致）
    # ====================================================================
    # 使用Hydra override来修改配置：
    # 1. runner.eval_policy_path: 设置为指定的policy路径
    # 2. env.eval.specific_reset_id: 设置为指定的task ID
    # 3. runner.only_eval: 设置为True，只进行评估不训练
    # 4. runner.logger.log_path: 设置为当前测试的日志目录
    # 5. env.seed: 如果提供了--seed参数，则设置为指定值
    CMD="python ${SRC_FILE} \
        --config-path ${EMBODIED_PATH}/config/ \
        --config-name ${BASE_CONFIG_NAME} \
        runner.eval_policy_path=\"${POLICY_PATH}\" \
        env.eval.specific_reset_id=${TASK_ID} \
        runner.only_eval=True \
        runner.logger.log_path=${TASK_RESULTS_DIR}"
    
    # 如果提供了seed参数，添加到命令中
    if [ -n "$ENV_SEED" ]; then
        CMD="${CMD} env.eval.seed=${ENV_SEED}"
    fi
    
    echo "执行命令:"
    echo "  ${CMD}"
    echo ""
    
    # ====================================================================
    # 执行评估
    # ====================================================================
    LOG_FILE="${TASK_RESULTS_DIR}/eval.log"
    
    # 运行评估命令，将输出保存到日志文件
    echo "正在运行评估..."
    if ${CMD} > "${LOG_FILE}" 2>&1; then
        echo "✓ Task ${TASK_ID} 评估成功完成"
        
        # 尝试从日志文件中提取成功率（根据实际日志格式调整）
        # 这里使用grep查找包含"success"的行，提取数字
        SUCCESS_RATE=$(grep -iE "success.*rate|success_rate|success rate|success.*:" "${LOG_FILE}" | tail -n 1 | grep -oE "[0-9]+\.[0-9]+|[0-9]+%" | head -n 1 || echo "N/A")
        
        # 记录成功任务
        SUCCESSFUL_TASKS+=("${TASK_ID}")
        TASK_RESULTS+=("Task ${TASK_ID}: SUCCESS (Success Rate: ${SUCCESS_RATE})")
        
        # 记录到汇总文件
        {
            echo "----------------------------------------"
            echo "Task ${TASK_ID} Result"
            echo "----------------------------------------"
            echo "Status: SUCCESS"
            echo "Success Rate: ${SUCCESS_RATE}"
            echo "Log File: ${LOG_FILE}"
            echo ""
        } >> "${SUMMARY_FILE}"
        
    else
        echo "✗ Task ${TASK_ID} 评估失败"
        
        # 记录失败任务
        FAILED_TASKS+=("${TASK_ID}")
        TASK_RESULTS+=("Task ${TASK_ID}: FAILED")
        
        # 记录到汇总文件
        {
            echo "----------------------------------------"
            echo "Task ${TASK_ID} Result"
            echo "----------------------------------------"
            echo "Status: FAILED"
            echo "Log File: ${LOG_FILE}"
            echo "请查看日志文件获取详细错误信息"
            echo ""
        } >> "${SUMMARY_FILE}"
    fi
    
    echo "日志文件: ${LOG_FILE}"
done

# ============================================================================
# 输出最终汇总
# ============================================================================

# 记录最终汇总到文件
{
    echo "=========================================="
    echo "Final Summary"
    echo "=========================================="
    echo "Total Tasks: ${#TASK_IDS[@]}"
    echo "Successful: ${#SUCCESSFUL_TASKS[@]}"
    echo "Failed: ${#FAILED_TASKS[@]}"
    echo ""
    if [ ${#SUCCESSFUL_TASKS[@]} -gt 0 ]; then
        echo "Successful Tasks: ${SUCCESSFUL_TASKS[@]}"
    fi
    if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
        echo "Failed Tasks: ${FAILED_TASKS[@]}"
    fi
    echo ""
    echo "End Time: $(date)"
    echo "=========================================="
} >> "${SUMMARY_FILE}"

# 输出最终汇总到控制台
echo ""
echo "=========================================="
echo "批量评估完成！"
echo "=========================================="
echo "总任务数: ${#TASK_IDS[@]}"
echo "成功: ${#SUCCESSFUL_TASKS[@]}"
echo "失败: ${#FAILED_TASKS[@]}"
echo ""

if [ ${#SUCCESSFUL_TASKS[@]} -gt 0 ]; then
    echo "成功的任务: ${SUCCESSFUL_TASKS[@]}"
fi

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "失败的任务: ${FAILED_TASKS[@]}"
fi

echo ""
echo "详细结果:"
for result in "${TASK_RESULTS[@]}"; do
    echo "  ${result}"
done

echo ""
echo "汇总文件: ${SUMMARY_FILE}"
echo "结果目录: ${RESULTS_BASE_DIR}"
echo "=========================================="

# 如果有失败的任务，返回非零退出码
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
