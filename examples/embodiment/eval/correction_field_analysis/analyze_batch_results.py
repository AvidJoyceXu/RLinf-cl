#!/usr/bin/env python3
"""
分析批量correction field analysis结果
扫描结果目录，提取统计信息，生成markdown表格和可视化
"""

import os
import re
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error: Required packages not found. Please install: numpy, pandas, matplotlib, seaborn")
    print(f"Missing package: {e}")
    sys.exit(1)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def parse_summary_file(summary_path: Path) -> Optional[Dict]:
    """
    解析summary_analysis.txt文件，提取关键信息
    
    Returns:
        dict: 包含task_i, task_j, direction_mean, scale_mean, dangerous_pct等信息
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return None
    
    result = {}
    
    # 提取Task Pair
    task_pair_match = re.search(r'Task Pair: Task (\d+) vs Task (\d+)', content)
    if task_pair_match:
        result['task_i'] = int(task_pair_match.group(1))
        result['task_j'] = int(task_pair_match.group(2))
    else:
        print(f"Warning: Could not parse task pair from {summary_path}")
        return None
    
    # 提取Direction Consistency Mean
    dir_mean_match = re.search(r'Direction Consistency:\s*\n\s*Mean: ([\d\.\-]+)', content)
    if dir_mean_match:
        result['direction_mean'] = float(dir_mean_match.group(1))
    else:
        print(f"Warning: Could not parse direction mean from {summary_path}")
        return None
    
    # 提取Scale Consistency Mean Log Scale
    scale_mean_match = re.search(r'Scale Consistency:\s*\n\s*Mean Log Scale: ([\d\.\-]+)', content)
    if scale_mean_match:
        result['scale_mean'] = float(scale_mean_match.group(1))
    else:
        print(f"Warning: Could not parse scale mean from {summary_path}")
        return None
    
    # 提取Dangerous States百分比
    dangerous_match = re.search(r'Dangerous States: \d+/\d+ \(([\d\.]+)%\)', content)
    if dangerous_match:
        result['dangerous_pct'] = float(dangerous_match.group(1))
    else:
        print(f"Warning: Could not parse dangerous percentage from {summary_path}")
        return None
    
    # 提取其他统计信息（可选）
    dir_std_match = re.search(r'Direction Consistency:\s*\n\s*Mean: [\d\.\-]+\s*\n\s*Std: ([\d\.\-]+)', content)
    if dir_std_match:
        result['direction_std'] = float(dir_std_match.group(1))
    
    scale_std_match = re.search(r'Scale Consistency:\s*\n\s*Mean Log Scale: [\d\.\-]+\s*\n\s*Std Log Scale: ([\d\.\-]+)', content)
    if scale_std_match:
        result['scale_std'] = float(scale_std_match.group(1))
    
    # 提取Merge Decision
    merge_match = re.search(r'Merge Decision:\s*\n\s*([✅❌]+)', content)
    if merge_match:
        result['can_merge'] = '✅' in merge_match.group(1)
    
    # 提取文件夹名称中的时间戳
    folder_name = summary_path.parent.name
    timestamp_match = re.search(r'(\d{8}_\d{6})$', folder_name)
    if timestamp_match:
        result['timestamp'] = timestamp_match.group(1)
    
    return result


def scan_results_directory(results_dir: Path) -> List[Dict]:
    """
    扫描结果目录，收集所有分析结果
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        list: 所有解析的结果字典列表
    """
    results = []
    
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return results
    
    # 查找所有correction_field_analysis_*格式的文件夹
    pattern = re.compile(r'correction_field_analysis_task\d+_task\d+_\d{8}_\d{6}')
    
    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue
        
        if not pattern.match(folder.name):
            continue
        
        summary_file = folder / "summary_analysis.txt"
        if not summary_file.exists():
            print(f"Warning: summary_analysis.txt not found in {folder}")
            continue
        
        parsed_result = parse_summary_file(summary_file)
        if parsed_result:
            parsed_result['folder'] = folder.name
            results.append(parsed_result)
    
    return results


def generate_markdown_table(results: List[Dict], output_path: Path):
    """
    生成markdown格式的统计表格
    
    Args:
        results: 解析的结果列表
        output_path: 输出文件路径
    """
    if not results:
        print("No results to generate table")
        return
    
    # 创建DataFrame以便于处理
    df = pd.DataFrame(results)
    
    # 按task_i和task_j排序
    df = df.sort_values(['task_i', 'task_j'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Correction Field Analysis Results Summary\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total pairs analyzed: {len(results)}\n\n")
        
        # 1. 完整结果表格
        f.write("## Complete Results Table\n\n")
        f.write("| Task I | Task J | Direction Mean | Scale Mean | Dangerous % | Can Merge |\n")
        f.write("|--------|--------|----------------|------------|-------------|-----------|\n")
        
        for _, row in df.iterrows():
            can_merge_str = "✅" if row.get('can_merge', False) else "❌"
            f.write(f"| {int(row['task_i'])} | {int(row['task_j'])} | "
                   f"{row['direction_mean']:.4f} | {row['scale_mean']:.4f} | "
                   f"{row['dangerous_pct']:.2f}% | {can_merge_str} |\n")
        
        f.write("\n")
        
        # 2. Direction统计
        f.write("## Direction Consistency Statistics\n\n")
        direction_values = df['direction_mean'].values
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {np.mean(direction_values):.4f} |\n")
        f.write(f"| Std | {np.std(direction_values):.4f} |\n")
        f.write(f"| Median | {np.median(direction_values):.4f} |\n")
        f.write(f"| Min | {np.min(direction_values):.4f} |\n")
        f.write(f"| Max | {np.max(direction_values):.4f} |\n")
        f.write("\n")
        
        # 3. Scale统计
        f.write("## Scale Consistency Statistics\n\n")
        scale_values = df['scale_mean'].values
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {np.mean(scale_values):.4f} |\n")
        f.write(f"| Std | {np.std(scale_values):.4f} |\n")
        f.write(f"| Median | {np.median(scale_values):.4f} |\n")
        f.write(f"| Min | {np.min(scale_values):.4f} |\n")
        f.write(f"| Max | {np.max(scale_values):.4f} |\n")
        f.write("\n")
        
        # 4. Dangerous统计
        f.write("## Dangerous States Statistics\n\n")
        dangerous_values = df['dangerous_pct'].values
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {np.mean(dangerous_values):.2f}% |\n")
        f.write(f"| Std | {np.std(dangerous_values):.2f}% |\n")
        f.write(f"| Median | {np.median(dangerous_values):.2f}% |\n")
        f.write(f"| Min | {np.min(dangerous_values):.2f}% |\n")
        f.write(f"| Max | {np.max(dangerous_values):.2f}% |\n")
        f.write("\n")
        
        # 5. Merge决策统计
        f.write("## Merge Decision Statistics\n\n")
        can_merge_count = df['can_merge'].sum() if 'can_merge' in df.columns else 0
        cannot_merge_count = len(df) - can_merge_count
        f.write(f"- Can Merge: {can_merge_count} ({can_merge_count/len(df)*100:.1f}%)\n")
        f.write(f"- Cannot Merge: {cannot_merge_count} ({cannot_merge_count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        # 6. 按Task分组统计
        f.write("## Statistics by Task\n\n")
        all_tasks = sorted(set(df['task_i'].tolist() + df['task_j'].tolist()))
        
        for task_id in all_tasks:
            task_rows = df[(df['task_i'] == task_id) | (df['task_j'] == task_id)]
            if len(task_rows) == 0:
                continue
            
            f.write(f"### Task {task_id}\n\n")
            f.write("| Metric | Direction Mean | Scale Mean | Dangerous % |\n")
            f.write("|--------|----------------|------------|-------------|\n")
            f.write(f"| Mean | {task_rows['direction_mean'].mean():.4f} | "
                   f"{task_rows['scale_mean'].mean():.4f} | "
                   f"{task_rows['dangerous_pct'].mean():.2f}% |\n")
            f.write(f"| Std | {task_rows['direction_mean'].std():.4f} | "
                   f"{task_rows['scale_mean'].std():.4f} | "
                   f"{task_rows['dangerous_pct'].std():.2f}% |\n")
            f.write("\n")
    
    print(f"Markdown table saved to: {output_path}")


def create_visualizations(results: List[Dict], output_dir: Path):
    """
    创建可视化图表
    
    Args:
        results: 解析的结果列表
        output_dir: 输出目录
    """
    if not results:
        print("No results to visualize")
        return
    
    df = pd.DataFrame(results)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 三个维度的分布直方图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Direction分布
    axes[0].hist(df['direction_mean'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(df['direction_mean'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["direction_mean"].mean():.4f}')
    axes[0].axvline(df['direction_mean'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["direction_mean"].median():.4f}')
    axes[0].set_xlabel('Direction Consistency Mean', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Direction Consistency Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scale分布
    axes[1].hist(df['scale_mean'], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1].axvline(df['scale_mean'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["scale_mean"].mean():.4f}')
    axes[1].axvline(df['scale_mean'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["scale_mean"].median():.4f}')
    axes[1].set_xlabel('Scale Consistency Mean (Log Scale)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Scale Consistency Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Dangerous分布
    axes[2].hist(df['dangerous_pct'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[2].axvline(df['dangerous_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["dangerous_pct"].mean():.2f}%')
    axes[2].axvline(df['dangerous_pct'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["dangerous_pct"].median():.2f}%')
    axes[2].set_xlabel('Dangerous States Percentage', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Dangerous States Distribution', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 热力图：Task Pair矩阵
    all_tasks = sorted(set(df['task_i'].tolist() + df['task_j'].tolist()))
    
    # 创建矩阵用于热力图
    for metric_name, metric_col in [('Direction', 'direction_mean'), 
                                     ('Scale', 'scale_mean'), 
                                     ('Dangerous', 'dangerous_pct')]:
        matrix = np.full((len(all_tasks), len(all_tasks)), np.nan)
        
        for _, row in df.iterrows():
            i_idx = all_tasks.index(int(row['task_i']))
            j_idx = all_tasks.index(int(row['task_j']))
            matrix[i_idx, j_idx] = row[metric_col]
            matrix[j_idx, i_idx] = row[metric_col]  # 对称矩阵
        
        # 对角线设为NaN（自己和自己不比较）
        np.fill_diagonal(matrix, np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdYlGn' if metric_name != 'Dangerous' else 'RdYlGn_r', 
                      aspect='auto', interpolation='nearest')
        
        # 设置刻度标签
        ax.set_xticks(range(len(all_tasks)))
        ax.set_yticks(range(len(all_tasks)))
        ax.set_xticklabels([f'Task {t}' for t in all_tasks])
        ax.set_yticklabels([f'Task {t}' for t in all_tasks])
        
        # 添加数值标注
        for i in range(len(all_tasks)):
            for j in range(len(all_tasks)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'{metric_name} Consistency Heatmap', fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_name.lower()}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. 散点图：Direction vs Scale，颜色表示Dangerous
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['direction_mean'], df['scale_mean'], 
                        c=df['dangerous_pct'], cmap='RdYlGn_r', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # 添加task pair标签
    for _, row in df.iterrows():
        ax.annotate(f"{int(row['task_i'])}-{int(row['task_j'])}", 
                   (row['direction_mean'], row['scale_mean']),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Direction Consistency Mean', fontsize=12)
    ax.set_ylabel('Scale Consistency Mean (Log Scale)', fontsize=12)
    ax.set_title('Direction vs Scale Consistency\n(Color: Dangerous States %)', 
                fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Dangerous States %')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'direction_vs_scale_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 箱线图：按Task分组
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 准备数据：每个task的所有pair值
    task_direction_data = []
    task_scale_data = []
    task_dangerous_data = []
    task_labels = []
    
    for task_id in all_tasks:
        task_rows = df[(df['task_i'] == task_id) | (df['task_j'] == task_id)]
        if len(task_rows) > 0:
            task_direction_data.append(task_rows['direction_mean'].values)
            task_scale_data.append(task_rows['scale_mean'].values)
            task_dangerous_data.append(task_rows['dangerous_pct'].values)
            task_labels.append(f'Task {task_id}')
    
    axes[0].boxplot(task_direction_data, labels=task_labels)
    axes[0].set_ylabel('Direction Consistency Mean', fontsize=12)
    axes[0].set_title('Direction Consistency by Task', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(task_scale_data, labels=task_labels)
    axes[1].set_ylabel('Scale Consistency Mean (Log Scale)', fontsize=12)
    axes[1].set_title('Scale Consistency by Task', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].boxplot(task_dangerous_data, labels=task_labels)
    axes[2].set_ylabel('Dangerous States %', fontsize=12)
    axes[2].set_title('Dangerous States by Task', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_by_task.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze batch correction field analysis results')
    parser.add_argument('--results_dir', type=str, 
                       default='/home/xulingyun/RLinf-cl/examples/embodiment/eval/results',
                       help='Directory containing analysis results')
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='Output directory for analysis results (default: results_dir/analysis_summary)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'analysis_summary'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning results directory: {results_dir}")
    results = scan_results_directory(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} analysis results")
    
    # 生成markdown表格
    markdown_path = output_dir / 'summary_statistics.md'
    generate_markdown_table(results, markdown_path)
    
    # 生成可视化
    create_visualizations(results, output_dir)
    
    # 保存原始数据为JSON
    json_path = output_dir / 'results_data.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Raw data saved to: {json_path}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
