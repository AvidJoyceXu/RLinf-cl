#!/bin/bash
# Auto-generated merge script
# Generated from: results/batch_eval_cross_task_20260123-15:41:27/evaluation_summary.txt
# Output directory: merged_policy_dir/libero_objectbash

set -e  # Exit on error

# Merge Task 1 and Task 8
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260122-13:30:27-libero_object_task1_lora_residual_sac_openvlaoft.yaml/libero_object_task1_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
                        /workspace/RLinf/logs/20260123-04:18:20-libero_object_task8_lora_residual_sac_openvlaoft.yaml/libero_object_task8_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/libero_object/task1_8/ 

# Merge Task 2 and Task 6
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260122-13:09:42-libero_object_task2_lora_residual_sac_openvlaoft.yaml/libero_object_task2_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260123-03:15:44-libero_object_task6_lora_residual_sac_openvlaoft.yaml/libero_object_task6_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/libero_objectbash/task2_6/ \
    --restore_norm

# Merge Task 2 and Task 9
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260122-13:09:42-libero_object_task2_lora_residual_sac_openvlaoft.yaml/libero_object_task2_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260123-11:48:31-libero_object_task9_lora_residual_sac_openvlaoft.yaml/libero_object_task9_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/libero_objectbash/task2_9/ \
    --restore_norm

# Merge Task 6 and Task 8
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260123-03:15:44-libero_object_task6_lora_residual_sac_openvlaoft.yaml/libero_object_task6_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260123-04:18:20-libero_object_task8_lora_residual_sac_openvlaoft.yaml/libero_object_task8_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/libero_objectbash/task6_8/ \
    --restore_norm

# Merge Task 7 and Task 8
python toolkits/merge_lora_policies/quick_merge.py \
    --checkpoint_paths /workspace/RLinf/logs/20260123-04:18:12-libero_object_task7_lora_residual_sac_openvlaoft.yaml/libero_object_task7_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model /workspace/RLinf/logs/20260123-04:18:20-libero_object_task8_lora_residual_sac_openvlaoft.yaml/libero_object_task8_single_trial_lora_openvlaoft/checkpoints/global_step_2000/actor/huggingface_model \
    --output_path merged_policy_dir/libero_objectbash/task7_8/ \
    --restore_norm

