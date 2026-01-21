#!/bin/bash

###############################################################################
# Batch Results Analysis Runner
# 
# 扫描correction field analysis结果目录，生成统计表格和可视化
###############################################################################

# =============================================================================
# 配置参数区域
# =============================================================================

# 结果目录（包含所有correction_field_analysis_*文件夹）
RESULTS_DIR="/workspace/RLinf/examples/embodiment/eval/results"

# 输出目录（可选，如果为空则使用results_dir/analysis_summary）
OUTPUT_DIR=""

# =============================================================================
# 脚本执行区域
# =============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "Batch Correction Field Analysis Results Analyzer"
echo "=============================================================================="
echo ""
echo "Results Directory: $RESULTS_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output Directory: $OUTPUT_DIR"
else
    echo "Output Directory: $RESULTS_DIR/analysis_summary"
fi
echo ""

# 构建Python命令
PYTHON_CMD="python analyze_batch_results.py"
PYTHON_CMD="$PYTHON_CMD --results_dir $RESULTS_DIR"

if [ -n "$OUTPUT_DIR" ]; then
    PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"
fi

echo "Running: $PYTHON_CMD"
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
