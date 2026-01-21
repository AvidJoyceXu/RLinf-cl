#!/bin/bash
# Auto-generated merge script
# Generated from: results/batch_eval_cross_task_20260121-10:39:25/evaluation_summary.txt
# Output directory: merged_policy_dir

set -e  # Exit on error

# Merge Task 0 and Task 2
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260121-07:24:09-libero_spatial_task0_lora_residual_sac_openvlaoft/task0_single_trial_lora_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260119-10:05:34-libero_spatial_task2_lora_residual_sac_openvlaoft/task2_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/task0_2/ \
    --restore_norm

# Merge Task 0 and Task 3
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260121-07:24:09-libero_spatial_task0_lora_residual_sac_openvlaoft/task0_single_trial_lora_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260119-13:18:01-libero_spatial_task3_lora_residual_sac_openvlaoft/task3_single_trial_residual_sac_openvlaoft/checkpoints/global_step_3000/actor/huggingface_model \
    --output_path merged_policy_dir/task0_3/ \
    --restore_norm

# Merge Task 2 and Task 6
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260119-10:05:34-libero_spatial_task2_lora_residual_sac_openvlaoft/task2_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260120-02:34:04-libero_spatial_task6_lora_residual_sac_openvlaoft/task6_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/task2_6/ \
    --restore_norm

# Merge Task 2 and Task 7
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260119-10:05:34-libero_spatial_task2_lora_residual_sac_openvlaoft/task2_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260120-02:34:18-libero_spatial_task7_lora_residual_sac_openvlaoft/task7_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/task2_7/ \
    --restore_norm

# Merge Task 3 and Task 7
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260119-13:18:01-libero_spatial_task3_lora_residual_sac_openvlaoft/task3_single_trial_residual_sac_openvlaoft/checkpoints/global_step_3000/actor/huggingface_model /workspace/RLinf/logs/20260120-02:34:18-libero_spatial_task7_lora_residual_sac_openvlaoft/task7_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/task3_7/ \
    --restore_norm

# Merge Task 4 and Task 6
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260119-16:57:44-libero_spatial_task4_lora_residual_sac_openvlaoft/task4_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260120-02:34:04-libero_spatial_task6_lora_residual_sac_openvlaoft/task6_single_trial_residual_sac_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/task4_6/ \
    --restore_norm

