# Residual SAC + RLinf README

## Commands
0. Deployment

```bash
docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --name rlinf \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   -v $RLINF_DIR:/workspace/RLinf \
   -v $HF_HOME:/workspace/hf \
   -v $LOG_DIR:/workspace/RLinf/logs \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

source switch_env openvla-oft
```

1. Training

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_task0_lora_residual_sac_openvlaoft
```

2. Eval single-task succ

```bash
bash examples/embodiment/eval_embodiment.sh libero_spatial_task0_lora_residual_sac_openvlaoft
```

3. Eval multi-task correction Field

```bash
bash examples/embodiment/eval/correction_field_analysis/run_correction_field_analysis.sh
```

在run_correction_field_analysis.sh脚本中，指定：
- `TASK_ID`
- checkpoint路径
- eval模式（base_rollout, demo, both_demo)

## Implementation

1. 模型架构
- `rlinf/models/embodiment/residual_policy/`文件夹

2. 模型配置
- `embodiment/config/model/residual_policy.yaml`
- `embodiment/config/model/lora_residual_policy.yaml`

3. 运行配置
- `embodiment/config/libero_spatial_task{$ID}_lora_residual_sac_openvlaoft.yaml`

4. 训练pipeline

- 计算rollout，填充replay buffer，传入rollout_batch
  - `rlinf/workers/rollout/hf/residual_rollout_worker.py`
  - 在`ChunkStepResult`中存储(obs, next_obs, base_action, base_next_action)
  - 特别地：访问base model，将当前obs的base action存入`last_forward_inputs`，存入rollout_batch

- 接收rollout_batch，从replay buffer中采样，训练actor和critic
  - `rlinf/workers/actor/residual_fsdp_sac_policy_worker.py`
  > 继承自`rlinf/workers/actor/fsdp_sac_policy_worker.py`
  > 重新实现了`forward_sac`, `forward_critic`等SAC相关方法
  > 不访问base model，直接从rollout_batch中读取base action

5. 数据通信

- rollout worker -> rollout_batch(`EmbodiedRolloutResult`) -> actor worker





