#!/usr/bin/env python3
"""
Generate merge commands based on cross-task evaluation results.

This script analyzes evaluation_summary.txt to find task pairs with high
bidirectional success rates and generates merge commands accordingly.
"""

import re
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List


def parse_evaluation_summary(summary_path: str) -> Tuple[Dict[int, str], Dict[Tuple[int, int], float]]:
    """
    Parse evaluation summary file to extract:
    1. Policy paths for each task
    2. Success rates for each (policy_task, eval_task) pair
    
    Returns:
        policy_paths: dict mapping task_id -> checkpoint_path
        success_rates: dict mapping (policy_task, eval_task) -> success_rate
    """
    policy_paths = {}
    success_rates = {}
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Extract policy paths
    policy_path_pattern = r'Task (\d+): (.+)'
    for match in re.finditer(policy_path_pattern, content):
        task_id = int(match.group(1))
        path = match.group(2).strip()
        if not path.startswith('/skip'):
            policy_paths[task_id] = path
    
    # Extract success rates
    result_pattern = r'Policy_Task_(\d+) -> Eval_Task_(\d+): SUCCESS \(Rate: ([\d.]+)\)'
    for match in re.finditer(result_pattern, content):
        policy_task = int(match.group(1))
        eval_task = int(match.group(2))
        rate = float(match.group(3))
        success_rates[(policy_task, eval_task)] = rate
    
    return policy_paths, success_rates


def find_good_pairs(success_rates: Dict[Tuple[int, int], float], 
                    min_bidirectional_rate: float = 0.5,
                    min_avg_rate: float = 0.65) -> List[Tuple[int, int]]:
    """
    Find task pairs with good bidirectional performance.
    
    Args:
        success_rates: dict mapping (policy_task, eval_task) -> success_rate
        min_bidirectional_rate: minimum rate for both directions
        min_avg_rate: minimum average rate across both directions
    
    Returns:
        List of (task_i, task_j) tuples where i < j
    """
    good_pairs = []
    processed = set()
    
    # Get all unique task pairs
    tasks = set()
    for (policy_task, eval_task) in success_rates.keys():
        tasks.add(policy_task)
        tasks.add(eval_task)
    
    for task_i in sorted(tasks):
        for task_j in sorted(tasks):
            if task_i >= task_j:
                continue
            
            pair = (task_i, task_j)
            if pair in processed:
                continue
            processed.add(pair)
            
            # Get bidirectional rates
            rate_ij = success_rates.get((task_i, task_j), 0.0)
            rate_ji = success_rates.get((task_j, task_i), 0.0)
            
            # Check if both directions meet minimum threshold
            if rate_ij >= min_bidirectional_rate and rate_ji >= min_bidirectional_rate:
                avg_rate = (rate_ij + rate_ji) / 2
                if avg_rate >= min_avg_rate:
                    good_pairs.append((task_i, task_j))
                    print(f"Found good pair ({task_i}, {task_j}): "
                          f"rate_{task_i}->{task_j}={rate_ij:.4f}, "
                          f"rate_{task_j}->{task_i}={rate_ji:.4f}, "
                          f"avg={avg_rate:.4f}")
    
    return good_pairs


def generate_merge_command(policy_paths: Dict[int, str], 
                          task_pair: Tuple[int, int],
                          output_base_dir: str = "merged_policy_dir") -> str:
    """
    Generate merge command for a task pair.
    
    Args:
        policy_paths: dict mapping task_id -> checkpoint_path
        task_pair: (task_i, task_j) tuple
        output_base_dir: base directory for merged policies
    
    Returns:
        Merge command string
    """
    task_i, task_j = task_pair
    
    if task_i not in policy_paths or task_j not in policy_paths:
        return None
    
    path_i = policy_paths[task_i]
    path_j = policy_paths[task_j]
    
    # Output directory: merged_policy_dir/task{i}_{j}/
    output_dir = f"{output_base_dir}/task{task_i}_{task_j}/"
    
    cmd = (
        f"python toolkits/merge_lora_policies/quick_merge.py \\\n"
        f"    --checkpoint_paths {path_i} {path_j} \\\n"
        f"    --output_path {output_dir} \\\n"
        f"    --restore_norm"
    )
    
    return cmd


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_merge_commands.py <evaluation_summary.txt> [output_base_dir]")
        sys.exit(1)
    
    summary_path = sys.argv[1]
    output_base_dir = sys.argv[2] if len(sys.argv) > 2 else "merged_policy_dir"
    
    if not os.path.exists(summary_path):
        print(f"Error: Evaluation summary file not found: {summary_path}")
        sys.exit(1)
    
    print(f"Parsing evaluation summary: {summary_path}")
    print("=" * 60)
    
    # Parse evaluation summary
    policy_paths, success_rates = parse_evaluation_summary(summary_path)
    
    print(f"\nFound {len(policy_paths)} policy paths:")
    for task_id in sorted(policy_paths.keys()):
        print(f"  Task {task_id}: {policy_paths[task_id]}")
    
    print(f"\nFound {len(success_rates)} success rate entries")
    
    # NOTE: 
    required_pairs = [(6, 8), (2, 9), (2, 6)]
    
    # Find good pairs automatically
    print("\n" + "=" * 60)
    print("Analyzing task pairs for good bidirectional performance...")
    print("=" * 60)
    good_pairs = find_good_pairs(success_rates, min_bidirectional_rate=0.1, min_avg_rate=0.5)
    
    # Combine required and good pairs (avoid duplicates)
    all_pairs = set(required_pairs)
    all_pairs.update(good_pairs)
    all_pairs = sorted(all_pairs)
    
    print(f"\n" + "=" * 60)
    print(f"Generating merge commands for {len(all_pairs)} task pairs:")
    print("=" * 60)
    
    # Generate commands
    commands = []
    for pair in all_pairs:
        cmd = generate_merge_command(policy_paths, pair, output_base_dir)
        if cmd:
            commands.append((pair, cmd))
            print(f"\n# Merge Task {pair[0]} and Task {pair[1]}")
            print(cmd)
        else:
            print(f"\n# Warning: Cannot generate command for pair {pair} (missing policy paths)")
    
    # Save to script file
    script_path = "merge_policies.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated merge script\n")
        f.write(f"# Generated from: {summary_path}\n")
        f.write(f"# Output directory: {output_base_dir}\n\n")
        f.write("set -e  # Exit on error\n\n")
        
        for pair, cmd in commands:
            f.write(f"# Merge Task {pair[0]} and Task {pair[1]}\n")
            f.write(cmd + "\n\n")
    
    os.chmod(script_path, 0o755)
    print(f"\n" + "=" * 60)
    print(f"✅ Generated merge script: {script_path}")
    print(f"   Run: bash {script_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
