#!/usr/bin/env python3
"""
分析correction field analysis结果，找出可以尝试合并的task组合
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

# 合并阈值（根据analyze_correction_vector_field.py中的阈值）
THRESHOLD_DIRECTION = 0.7  # 方向一致性阈值
THRESHOLD_SCALE = 1.0       # 尺度一致性阈值
THRESHOLD_DANGEROUS = 5.0   # 危险状态百分比阈值（越低越好）


def load_results(json_path: Path) -> List[Dict]:
    """加载结果数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_merge_candidates(results: List[Dict]) -> Dict:
    """分析可以合并的候选组合"""
    
    # 分类组合
    candidates = {
        'excellent': [],      # 所有指标都接近阈值
        'good_direction': [],  # Direction较高（>0.3）
        'low_dangerous': [],   # Dangerous较低（<60%）
        'good_scale': [],      # Scale较高（>0.6）
        'promising': [],       # 综合评分较高
        'poor': []             # 所有指标都很差
    }
    
    for result in results:
        task_i = result['task_i']
        task_j = result['task_j']
        direction = result['direction_mean']
        scale = result['scale_mean']
        dangerous = result['dangerous_pct']
        
        # 计算综合评分（归一化到0-1）
        # Direction: 越高越好，归一化到[0,1]（假设范围是[-1,1]）
        dir_score = (direction + 1) / 2
        
        # Scale: 越高越好，归一化到[0,1]（假设范围是[0,1]）
        scale_score = min(scale / THRESHOLD_SCALE, 1.0)
        
        # Dangerous: 越低越好，归一化到[0,1]（假设范围是[0,100]）
        dangerous_score = 1.0 - (dangerous / 100.0)
        
        # 综合评分（加权平均）
        composite_score = (dir_score * 0.4 + scale_score * 0.3 + dangerous_score * 0.3)
        
        pair_info = {
            'task_i': task_i,
            'task_j': task_j,
            'direction': direction,
            'scale': scale,
            'dangerous': dangerous,
            'composite_score': composite_score,
            'dir_score': dir_score,
            'scale_score': scale_score,
            'dangerous_score': dangerous_score
        }
        
        # 分类
        if direction >= THRESHOLD_DIRECTION * 0.8 and dangerous <= THRESHOLD_DANGEROUS * 10:
            candidates['excellent'].append(pair_info)
        elif direction >= 0.3:
            candidates['good_direction'].append(pair_info)
        elif dangerous <= 60.0:
            candidates['low_dangerous'].append(pair_info)
        elif scale >= 0.6:
            candidates['good_scale'].append(pair_info)
        elif composite_score >= 0.5:
            candidates['promising'].append(pair_info)
        else:
            candidates['poor'].append(pair_info)
    
    # 按综合评分排序
    for key in candidates:
        candidates[key].sort(key=lambda x: x['composite_score'], reverse=True)
    
    return candidates


def print_analysis(candidates: Dict):
    """打印分析结果"""
    
    print("=" * 80)
    print("Correction Field Analysis - Merge Candidates")
    print("=" * 80)
    print("\n合并阈值:")
    print(f"  Direction: >= {THRESHOLD_DIRECTION}")
    print(f"  Scale: >= {THRESHOLD_SCALE}")
    print(f"  Dangerous: <= {THRESHOLD_DANGEROUS}%")
    print("\n" + "=" * 80)
    
    # 1. 优秀候选（最接近阈值）
    if candidates['excellent']:
        print("\n🌟 优秀候选（最接近合并阈值）:")
        print("-" * 80)
        print(f"{'Task Pair':<15} {'Direction':<12} {'Scale':<12} {'Dangerous':<12} {'综合评分':<10}")
        print("-" * 80)
        for pair in candidates['excellent']:
            print(f"Task {pair['task_i']}-{pair['task_j']:<8} "
                  f"{pair['direction']:>10.4f}  {pair['scale']:>10.4f}  "
                  f"{pair['dangerous']:>9.2f}%  {pair['composite_score']:>8.3f}")
    
    # 2. 方向一致性较好的组合
    if candidates['good_direction']:
        print("\n📈 方向一致性较好的组合 (Direction >= 0.3):")
        print("-" * 80)
        print(f"{'Task Pair':<15} {'Direction':<12} {'Scale':<12} {'Dangerous':<12} {'综合评分':<10}")
        print("-" * 80)
        for pair in candidates['good_direction'][:5]:  # 只显示前5个
            print(f"Task {pair['task_i']}-{pair['task_j']:<8} "
                  f"{pair['direction']:>10.4f}  {pair['scale']:>10.4f}  "
                  f"{pair['dangerous']:>9.2f}%  {pair['composite_score']:>8.3f}")
    
    # 3. 危险状态较少的组合
    if candidates['low_dangerous']:
        print("\n✅ 危险状态较少的组合 (Dangerous <= 60%):")
        print("-" * 80)
        print(f"{'Task Pair':<15} {'Direction':<12} {'Scale':<12} {'Dangerous':<12} {'综合评分':<10}")
        print("-" * 80)
        for pair in candidates['low_dangerous']:
            print(f"Task {pair['task_i']}-{pair['task_j']:<8} "
                  f"{pair['direction']:>10.4f}  {pair['scale']:>10.4f}  "
                  f"{pair['dangerous']:>9.2f}%  {pair['composite_score']:>8.3f}")
    
    # 4. 尺度一致性较好的组合
    if candidates['good_scale']:
        print("\n📏 尺度一致性较好的组合 (Scale >= 0.6):")
        print("-" * 80)
        print(f"{'Task Pair':<15} {'Direction':<12} {'Scale':<12} {'Dangerous':<12} {'综合评分':<10}")
        print("-" * 80)
        for pair in candidates['good_scale']:
            print(f"Task {pair['task_i']}-{pair['task_j']:<8} "
                  f"{pair['direction']:>10.4f}  {pair['scale']:>10.4f}  "
                  f"{pair['dangerous']:>9.2f}%  {pair['composite_score']:>8.3f}")
    
    # 5. 有潜力的组合（综合评分较高）
    if candidates['promising']:
        print("\n💡 有潜力的组合（综合评分 >= 0.5）:")
        print("-" * 80)
        print(f"{'Task Pair':<15} {'Direction':<12} {'Scale':<12} {'Dangerous':<12} {'综合评分':<10}")
        print("-" * 80)
        for pair in candidates['promising'][:10]:  # 显示前10个
            print(f"Task {pair['task_i']}-{pair['task_j']:<8} "
                  f"{pair['direction']:>10.4f}  {pair['scale']:>10.4f}  "
                  f"{pair['dangerous']:>9.2f}%  {pair['composite_score']:>8.3f}")
    
    # 6. 推荐合并的组合
    print("\n" + "=" * 80)
    print("🎯 推荐尝试合并的组合（按优先级排序）:")
    print("=" * 80)
    
    # 合并所有候选并按综合评分排序
    all_candidates = []
    for category in ['excellent', 'low_dangerous', 'good_direction', 'promising']:
        all_candidates.extend(candidates[category])
    
    # 去重（保留评分最高的）
    seen_pairs = {}
    for pair in all_candidates:
        pair_key = tuple(sorted([pair['task_i'], pair['task_j']]))
        if pair_key not in seen_pairs or pair['composite_score'] > seen_pairs[pair_key]['composite_score']:
            seen_pairs[pair_key] = pair
    
    recommended = sorted(seen_pairs.values(), key=lambda x: x['composite_score'], reverse=True)
    
    print(f"\n{'排名':<6} {'Task Pair':<15} {'Direction':<12} {'Scale':<12} {'Dangerous':<12} {'综合评分':<10} {'推荐理由':<30}")
    print("-" * 100)
    
    for idx, pair in enumerate(recommended[:10], 1):
        reasons = []
        if pair['direction'] >= 0.3:
            reasons.append("方向较好")
        if pair['dangerous'] <= 60:
            reasons.append("危险度低")
        if pair['scale'] >= 0.5:
            reasons.append("尺度较好")
        reason_str = ", ".join(reasons) if reasons else "综合评分较高"
        
        print(f"{idx:<6} Task {pair['task_i']}-{pair['task_j']:<8} "
              f"{pair['direction']:>10.4f}  {pair['scale']:>10.4f}  "
              f"{pair['dangerous']:>9.2f}%  {pair['composite_score']:>8.3f}  {reason_str:<30}")
    
    print("\n" + "=" * 80)
    print("注意事项:")
    print("1. 所有组合目前都未达到严格的合并阈值（Direction >= 0.7, Dangerous <= 5%）")
    print("2. 推荐组合是基于相对较好的指标，但仍需谨慎评估")
    print("3. 建议先在小规模测试中验证合并效果")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze merge candidates from correction field analysis')
    parser.add_argument('--json_file', type=str,
                       default='/workspace/RLinf/examples/embodiment/eval/results/libero_object/analysis_summary/results_data.json',
                       help='Path to results_data.json file')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    results = load_results(json_path)
    candidates = analyze_merge_candidates(results)
    print_analysis(candidates)


if __name__ == "__main__":
    main()
