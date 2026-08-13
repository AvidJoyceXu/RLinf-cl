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

---

## CoRL 2026 rebuttal experiments (RGB observation)

环境搭建、踩过的坑、故障排查见 **`SETUP.md`**；实验结果与结论见 **`rebuttal/results-rgb-merge.md`**；
baseline 调研见 **`rebuttal/baselines/baseline-survey.md`**。下面只列复现命令。

所有命令在容器内 `/workspace/RLinf` 下执行。⚠️ `docker exec ... bash -lc` **不会**激活 venv，
启动任何脚本都要显式 `export PATH=/opt/venv/openvla-oft/bin:$PATH`。
⚠️ 并行跑多个 run 必须**一个容器一个 run**，否则会共用同一个 Ray cluster 而互相拖垮/连坐（`SETUP.md` §7.1b）。

### 1. RGB observation 模式

去掉 privileged 的 object-to-eef 相对位姿，改成与真机同构的输入：

```
rl_flatten_obs = [f(I_{t-1}), f(I_t), p_{t-1}, p_t]      # 846 维
  f = 冻结 DINOv2-small CLS 特征（384 维，单个第三人称相机）
  p = robot0_proprio-state（39 维）
```

由 `env.{train,eval}.obs_mode: rgb` 开启，默认仍是 `privileged`，老 config 与老 checkpoint 不受影响。
两种模式的 checkpoint **不可互换**（`fc1_A` 输入宽度 846 vs 88/74）。

```bash
# 生成 RGB config（从 privileged config 派生，只改观测通路，其余超参逐字不变）
python rebuttal/make_rgb_configs.py --suite libero_object  --tasks 1 6 7 8 9
python rebuttal/make_rgb_configs.py --suite libero_spatial --tasks 0 2 3 6 7

# 10 个单任务 expert，一容器一 run，各绑一张卡
bash rebuttal/launch_rgb_runs.sh
```

正文用的是 `global_step_1000` 的 checkpoint（SR 在 800–1000 步已饱和，见 `results-rgb-merge.md` §1）。

### 2. 跨任务 SR 矩阵

```bash
bash rebuttal/run_cross_task_sr.sh <gpu> <tag> <config> P1=<ckpt> P6=<ckpt> ... -- 1 6 7 8 9
```

⚠️ 该脚本的进度回显在评测失败时**同样打印 `SR=`**（值为空），不能用来判断完成度；
判断完成必须查 `results/<tag>/<name>_on_t<task>.log` 里是否真有 `eval/success_once`。

frozen base 下界用 `residual_policy.res_scale=0`：

```bash
bash rebuttal/run_eval.sh _run_rgb_t1 0 base_t1 \
  runner.eval_policy_path=<any_846d_ckpt> residual_policy.res_scale=0 env.eval.specific_reset_id=1
```

### 3. RFC（correction field）

```bash
bash rebuttal/run_rfc_pairs.sh <gpu> <tag> eval_lora_config_object <ckpt_root> 1:6 1:7 ...
```

RFC 的定义与正文一致：**RFC = DC · MC**，其中 DC 是两个 residual 修正的余弦相似度、
MC = exp(−|log(‖a_i‖/‖a_j‖)|) ∈ (0,1] 是幅度一致性，均在**第一个 action chunk（7 维）**上、
乘 `res_scale` 之前取值，因此 RFC 与 α 无关。probe state 用 `--num_probe_states 2000`
固定每个任务的样本量，保证矩阵内各对可比。

⚠️ 单次运行内会打印 `Overall Direction Consistency: <mean> ± <std>`，那个 `±` 是**跨 probe state**
的离散度，不是重复测量的误差；同一 checkpoint 重复运行的结果是**逐位相同**的（已验证）。

### 4. τ sweep、merge 与合并后评测

```bash
# 贪心 replay：给定 RFC 矩阵与到达顺序，解出各 τ 下的 expert bank
python rebuttal/rfc_merge_sweep.py --rfc results/rfc_rgbfixed --order 1 6 7 8 9 \
  --extra-cache <merged_expert_rfc.json> --tau-grid 0.25 0.20 0.16 0.10 0.05 0.0

# 合并（按论文语义是增量合并：把新任务并进选中的 expert，而非多个单任务一次性平均）
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths <merged_1_6> <task8> --output_path <merged_1_6_8> --restore_norm
```

merge 算子对照（rebuttal 用）：

```bash
python rebuttal/ties_merge.py --checkpoints <ckpt_i> <ckpt_j> --output <out> --k 0.2
```

### 5. Merge 后 refine

从合并 checkpoint 热启动一个新 run（新 optimizer、新 replay buffer），在该 expert 覆盖的任务上继续 RL：

```yaml
actor.model.model_path: <merged_ckpt_dir>   # 见下方「权重加载」
env.train.specific_reset_id: [6, 7]         # 该 expert 覆盖的任务，支持列表
runner.max_epochs: 300
```

单个 global step 的环境交互实测：`32 envs × 16 env-steps = 512` 环境步、64 transitions。

### 6. ⚠️ 权重加载（这是一处已修复的严重 bug）

`rlinf/models/embodiment/residual_policy/get_model()` 原先**只按 config 构造随机初始化的模型，
完全忽略 `cfg.model_path`**。调用方设置了 `model_path` 并打印「加载成功」，实际权重一个都没进去。
所有经由该路径的 RFC 数值都是在**两个随机初始化的策略**上算出来的，因此才会出现同一对任务重复测量
得到 −0.24 ~ 0.63 的现象。

现在 `model_path` 非空即真正加载 safetensors 并做严格校验（`q_head` 允许缺失，
它只在训练期由 critic 使用）；`model_path` 为空时行为不变，仍是随机初始化，训练侧不受影响。

自查工具：

```bash
python rebuttal/residual_load_determinism.py --checkpoint <ckpt> --obs_dim 846
# 期望 "all parameters identical" + mean cosine(load1, load2) = 1.0
python rebuttal/probe_state_determinism.py --task 1 --episodes 2 --obs_mode rgb
# 期望 "byte-identical: True"
```





