#!/bin/bash

###############################################################################
# 整理correction field analysis结果文件夹
# 根据日期将文件夹分类到不同的子目录中
###############################################################################

# =============================================================================
# 配置参数区域
# =============================================================================

# 要扫描的目录
SCAN_DIR="/workspace/RLinf/examples/embodiment/eval/results/"

# 目标目录（如果为空，则在SCAN_DIR下创建）
TARGET_BASE_DIR=""

# 日期到文件夹名称的映射
# 格式：日期:文件夹名
declare -A DATE_TO_FOLDER
DATE_TO_FOLDER["20260121"]="libero_spatial"
DATE_TO_FOLDER["20260123"]="libero_object"

# =============================================================================
# 脚本执行区域
# =============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确定目标基础目录
if [ -z "$TARGET_BASE_DIR" ]; then
    TARGET_BASE_DIR="$SCAN_DIR"
fi

echo "=============================================================================="
echo "Organizing Correction Field Analysis Results by Date"
echo "=============================================================================="
echo ""
echo "Scan Directory: $SCAN_DIR"
echo "Target Base Directory: $TARGET_BASE_DIR"
echo ""

# 检查扫描目录是否存在
if [ ! -d "$SCAN_DIR" ]; then
    echo "❌ Error: Scan directory does not exist: $SCAN_DIR"
    exit 1
fi

# 统计信息
declare -A MOVED_COUNT
declare -A SKIPPED_COUNT
TOTAL_SCANNED=0

# 遍历所有子文件夹
for folder in "$SCAN_DIR"/*; do
    # 检查是否是目录
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    TOTAL_SCANNED=$((TOTAL_SCANNED + 1))
    
    # 检查文件夹名是否包含日期
    matched_date=""
    target_folder=""
    
    for date_pattern in "${!DATE_TO_FOLDER[@]}"; do
        if [[ "$folder_name" == *"$date_pattern"* ]]; then
            matched_date="$date_pattern"
            target_folder="${DATE_TO_FOLDER[$date_pattern]}"
            break
        fi
    done
    
    if [ -z "$matched_date" ]; then
        echo "⚠️  Skipping: $folder_name (no matching date pattern)"
        SKIPPED_COUNT["other"]=$((${SKIPPED_COUNT["other"]:-0} + 1))
        continue
    fi
    
    # 创建目标目录
    target_path="$TARGET_BASE_DIR/$target_folder"
    mkdir -p "$target_path"
    
    # 移动文件夹
    target_full_path="$target_path/$folder_name"
    
    # 检查目标是否已存在
    if [ -e "$target_full_path" ]; then
        echo "⚠️  Warning: Target already exists, skipping: $target_full_path"
        SKIPPED_COUNT["$matched_date"]=$((${SKIPPED_COUNT["$matched_date"]:-0} + 1))
        continue
    fi
    
    # 执行移动
    if mv "$folder" "$target_full_path" 2>/dev/null; then
        echo "✅ Moved: $folder_name -> $target_folder/"
        MOVED_COUNT["$matched_date"]=$((${MOVED_COUNT["$matched_date"]:-0} + 1))
    else
        echo "❌ Error: Failed to move $folder_name"
        SKIPPED_COUNT["$matched_date"]=$((${SKIPPED_COUNT["$matched_date"]:-0} + 1))
    fi
done

# 打印统计信息
echo ""
echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Total folders scanned: $TOTAL_SCANNED"
echo ""

echo "Moved by date:"
for date_pattern in "${!DATE_TO_FOLDER[@]}"; do
    folder_name="${DATE_TO_FOLDER[$date_pattern]}"
    count=${MOVED_COUNT["$date_pattern"]:-0}
    echo "  $date_pattern -> $folder_name/: $count folders"
done

echo ""
echo "Skipped:"
for key in "${!SKIPPED_COUNT[@]}"; do
    count=${SKIPPED_COUNT[$key]}
    echo "  $key: $count folders"
done

echo ""
echo "=============================================================================="
echo "✅ Organization complete!"
echo "=============================================================================="
