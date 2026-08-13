# 环境搭建与实验运行指南（LIBERO / OpenVLA-OFT residual SAC）

面向：拿到一台新机器、或者第一次接手这套仿真实验的人。

本文里的命令都在 `rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0` 镜像里实际跑过。凡是标 ⚠️ 的都是踩过的坑，不看会浪费时间。

---

## 1. 需要准备的东西

| 类别 | 内容 | 大小 |
|---|---|---|
| Docker 镜像 | `rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0` | ~50 GB |
| Base policy（必需） | `Haozhan72/Openvla-oft-SFT-libero-object-traj1`<br>`Haozhan72/Openvla-oft-SFT-libero-spatial-traj1` | 各 ~15 GB |
| 视觉编码器（RGB 模式必需） | `facebook/dinov2-small` | ~90 MB |
| Task-expert residual checkpoint（复现 merge/RFC 用） | `AvidJoyce/icml-2025-checkpoints` 下的 `task-expert/` | 每个 12 KB |
| GPU | 单机多卡；单卡至少 ~25 GB 空闲（7B base bf16 + 环境渲染） | — |
| 磁盘 | 镜像 + checkpoint + logs，建议 ≥ 200 GB | — |

⚠️ **`HF_HOME` 和 docker root dir 在本项目里都不是默认值**。当前机器上是：

```
HF_HOME=/data2/joycexu/.cache/huggingface
Docker Root Dir=/data2/docker      # docker info | grep "Root Dir"
```

换机器时先确认这两个位置在大盘上，否则系统盘会被撑爆。

---

## 2. 启动容器

```bash
export RLINF_DIR=/data2/joycexu/RLinf-cl                 # 本仓库
export HF_HOME=/data2/joycexu/.cache/huggingface         # 模型缓存
export LOG_DIR=$RLINF_DIR/logs                           # 训练日志（可以指向别的大盘）

docker run -d --gpus all \
  --shm-size 100g \
  --net=host \
  --name rlinf \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/workspace/hf \
  -v $RLINF_DIR:/workspace/RLinf \
  -v $HF_HOME:/workspace/hf \
  -v $LOG_DIR:/workspace/RLinf/logs \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 sleep infinity

docker exec -it rlinf bash
```

⚠️ **容器里默认激活的是 `openvla` 环境，不是 `openvla-oft`**（`/root/.bashrc` 最后一行写死了 `source /opt/venv/openvla/bin/activate`）。进容器后必须切：

```bash
source switch_env openvla-oft
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# 期望：2.6.0+cu124 4.40.1
```

镜像里有 4 个 venv：`/opt/venv/{gr00t,openpi,openvla,openvla-oft}`。**LIBERO + OpenVLA-OFT 的实验一律用 `openvla-oft`。**
不想 `source` 的场合（比如脚本里）直接用绝对路径：`/opt/venv/openvla-oft/bin/python`。

### 2.1 ⚠️ 联网（本机需要代理）

当前机器走本地代理 `http://127.0.0.1:7891`。容器用 `--net=host` 共享网络命名空间，但**不会继承宿主机的代理环境变量**，需要显式传：

```bash
docker exec -e https_proxy=http://127.0.0.1:7891 -e http_proxy=http://127.0.0.1:7891 rlinf bash
```

不传的话表现是 `OSError: We couldn't connect to 'https://huggingface.co'`。
模型下载完一次之后落进挂载的 `HF_HOME`，**后续训练/评测不再需要联网**。

---

## 3. 准备模型资产

在容器里（已切 `openvla-oft`、已配代理）：

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ["Haozhan72/Openvla-oft-SFT-libero-object-traj1",
             "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1",
             "facebook/dinov2-small"]:
    print(repo, snapshot_download(repo))
EOF
```

下完之后确认路径，**config 里 `residual_policy.base_model_path` 和 `actor.base_model.model_path` 写的是 snapshot 的绝对路径**，换机器后必须改：

```bash
ls /workspace/hf/hub/models--Haozhan72--Openvla-oft-SFT-libero-object-traj1/snapshots/
# 当前 config 里写死的是 62e5a8daba3f619c993f23b248ae04b0d9677bb5（object）
#                        39e5240e879c80b6cda6b3a83763dad717f5c05d（spatial）
```

Task-expert residual checkpoint（RFC / merge 实验用，不训练的话也需要）：

```
$HF_HOME/hub/models--AvidJoyce--icml-2025-checkpoints/snapshots/<rev>/task-expert/
  libero-object/task{1,2,6,7,8,9}/model-00001-of-00001.safetensors
  libero-spatial/task{0,2,3,4,6,7}/model-00001-of-00001.safetensors
```

⚠️ 正文用的任务集是 **object `[1,6,7,8,9]`、spatial `[0,2,3,6,7]`**；上面多出来的 object-task2 和 spatial-task4 不进主表。

---

## 4. 验证安装

跑一遍 smoke test（不需要 base model，2 分钟内完成，会拉一次 dinov2）：

```bash
docker exec -w /workspace/RLinf \
  -e https_proxy=http://127.0.0.1:7891 -e http_proxy=http://127.0.0.1:7891 rlinf bash -lc '
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  export PYTHONPATH=/workspace/RLinf EMBODIED_PATH=/workspace/RLinf/examples/embodiment
  /opt/venv/openvla-oft/bin/python rebuttal/smoke_test_rgb_obs.py'
```

期望输出：

```
[ok] frame stacking across reset boundaries
[ok] frozen visual encoder (feature_dim=384, obs_dim=846, device=cuda)
[ok] rollout worker visual encoding (order, idempotence, passthrough)
[ok] LIBERO env end-to-end in rgb mode (obs_dim=846, auto-reset exercised)
[ok] privileged mode unregressed (obs_dim=88)
all smoke tests passed
```

其中第 4、5 项会真的起 LIBERO 环境，能跑通就说明 mujoco / robosuite / LIBERO 资产都是好的。

---

## 5. 跑实验

以下都在容器里、`/workspace/RLinf` 目录下、已切 `openvla-oft`。

### 5.1 单任务 residual RL 训练

```bash
bash examples/embodiment/run_embodiment.sh libero_object_task1_lora_residual_sac_openvlaoft
```

- config 名 = `examples/embodiment/config/<name>.yaml` 去掉后缀。
- 日志和 checkpoint 落在 `logs/<时间戳>-<config名>/<experiment_name>/checkpoints/global_step_*/actor/huggingface_model`。
- 训练脚本自己会 `export MUJOCO_GL=egl`、`PYTHONPATH`、`EMBODIED_PATH`，不用手动设。
- 正文用的是 `global_step_2000`（个别任务 3000）。

⚠️ **占几张卡 / 怎么限制**：config 里 `cluster.component_placement: {actor,env,rollout: all}` 会吃掉节点上的**全部** GPU。

**`CUDA_VISIBLE_DEVICES` 不起作用**——调度器自己探测节点硬件，设了也照样看到 8 张卡（日志里会打印 `Using flexible placement with hardware ranks: [[0],...,[7]]`）。唯一的办法是改 config 里的 placement：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: "0"      # 只用 0 号卡；"0-3" = 用 0~3 号卡
```

这个 key 带逗号，**没法用 hydra 命令行 override**（`++cluster.component_placement={"actor,env,rollout":"0"}` 会报 `no viable alternative at input`）。要临时限卡就复制一份 config 改掉：

```bash
sed 's/^    actor,env,rollout: all$/    actor,env,rollout: "0"/' \
  examples/embodiment/config/libero_object_task1_lora_residual_sac_openvlaoft_rgb.yaml \
  > examples/embodiment/config/_tmp_1gpu.yaml
```

⚠️ 改了卡数就要同步改 batch：`actor.global_batch_size` 必须能被 `actor.micro_batch_size × actor_world_size`（= GPU 数）整除，否则启动时直接 assert 失败。

显存不够时可调的旋钮，按影响从小到大：`env.train.total_num_envs`（64→32→16）、`actor.enable_offload=True`、`rollout.enable_offload=True`。

**实测显存占用**（H20，bf16，`rebuttal/probe_gpu_footprint.py` 测的）：

| 项目 | 显存 |
|---|---|
| OpenVLA-OFT base model 权重 | 14.1 GiB |
| + 冻结 ViT + residual policy | 14.2 GiB（residual 只加 0.09） |
| rollout batch 8（峰值） | 18.7 GiB |
| rollout batch 16（峰值） | 23.3 GiB |
| rollout batch 32 | OOM（该卡当时空闲 ~30 GiB） |

也就是**每张卡至少要留 ~20 GiB**（对应每卡 8 个 env）。

### 5.2 单任务评测

```bash
# 用 config 里的 eval_policy_path
bash examples/embodiment/eval_embodiment.sh libero_object_task1_lora_residual_sac_openvlaoft

# 指定 policy 和 task id（补测常用）
bash examples/embodiment/eval_single_task.sh \
  libero_object_task1_lora_residual_sac_openvlaoft \
  /workspace/RLinf/logs/<...>/checkpoints/global_step_2000/actor/huggingface_model \
  1
```

结果在 `results/single_eval_<时间戳>/`。

### 5.3 跨任务 SR 矩阵

```bash
bash examples/embodiment/batch_eval_cross_task.sh <config_name> <path_task0> <path_task1> ... <path_task9>
# 用不到的槽位填 /skip
```

⚠️ **脚本里 `SKIP_TASKS` 是硬编码的**（`batch_eval_cross_task.sh:76`，当前是 `(0 3 4 5)`，即保留 1,2,6,7,8,9 —— 对应 libero_object）。**文件头注释写的是「跳过 1,8,9」，和代码不一致，以代码为准。** 换 suite 时必须改这一行，否则会正好跳掉你要测的任务。

输出 `results/batch_eval_cross_task_<时间戳>/evaluation_summary.txt`。

### 5.4 RFC / correction field 分析

```bash
cd examples/embodiment/eval/correction_field_analysis
bash run_correction_field_analysis.sh              # libero_spatial
bash run_correction_field_analysis_libero_object.sh # libero_object
```

参数在脚本顶部改：`CHECKPOINT_PATHS[task_id]`、`SKIP_TASKS`、`STATE_METHOD`、`MAX_DEMOS`、`DELTA_DIR`。

单对分析：

```bash
python analyze_correction_vector_field.py \
  --config configs/eval_lora_config.yaml \
  --task_i 6 --task_j 8 \
  --checkpoint_i <path> --checkpoint_j <path> \
  --state_method base_rollout --max_demos 5
```

Probe state 协议（**正文用的就是这个**）：`base_rollout`，两个任务各用 frozen base policy rollout 5 个 episode（每个 ≤240 步），状态取并集，N 通常落在 2000 附近。
⚠️ `demo` / `both_demo` 模式需要 LIBERO 的 hdf5 数据集，`base_rollout` 不需要。

⚠️ **脚本输出的 `can_merge` 不能直接用**：`merge_decision()` 里的阈值是硬编码的 `dir=0.7 / scale=1.0 / p_bad=0.05`，且没有暴露成命令行参数。而 sim 上 DC 最高只有 ~0.39，按 0.7 没有任何一对会通过。正文实际用的阈值是 **libero_object τ=0.1、libero_spatial τ=0.2**，是人工看 `Direction Mean` 那一列定的。要复现主表结果就看 `summary_analysis.txt` 里的 `Mean` 自己比阈值。

### 5.5 Merge 与合并后评测

```bash
python toolkits/merge_lora_policies/quick_merge.py \
  --checkpoint_paths <path_task_i> <path_task_j> \
  --output_path merged_policy_dir/libero_object/task6_8/ \
  --restore_norm

bash examples/embodiment/eval_single_task.sh <config_name> merged_policy_dir/libero_object/task6_8 6
# 或批量：
bash eval_merged_policies.sh <config_name> merged_policy_dir
```

⚠️ `output_path` 结尾**不是** `.pt` 就存 safetensors（RLinf checkpoint 格式），是 `.pt` 就存 PyTorch 格式。评测要用 safetensors 那种。

⚠️ merge-only 的效果通常明显低于单任务 policy，需要 post-merge refine（正文 object 用了 600 steps）才能恢复。报结果时 merge-only 和 merge+refine 要分开列。

---

## 6. RGB observation 模式（新增）

用来去掉 privileged 的 object-to-eef 相对位姿，改成和真机同构的视觉输入。

```
rl_flatten_obs = [f(I_{t-1}), f(I_t), p_{t-1}, p_t]
                 f = 冻结 DINOv2-small CLS 特征（384 维，单个第三人称相机）
                 p = robot0_proprio-state（39 维）
                 -> 846 维
```

跑法和 privileged 完全一样，只是换 config：

```bash
bash examples/embodiment/run_embodiment.sh libero_object_task1_lora_residual_sac_openvlaoft_rgb
```

RFC 分析要加 `--obs_mode rgb`（否则 probe state 是 88 维、和 RGB checkpoint 对不上）：

```bash
python analyze_correction_vector_field.py --config configs/eval_lora_config.yaml \
  --task_i 6 --task_j 8 --checkpoint_i <rgb_ckpt> --checkpoint_j <rgb_ckpt> \
  --obs_mode rgb --state_method base_rollout
```

给别的 suite 生成 RGB config：

```bash
python rebuttal/make_rgb_configs.py --suite libero_spatial --tasks 0 2 3 6 7
```

这个脚本从 privileged config 派生，只改 5 处（backbone 引用、experiment_name、`obs_mode`×2、`obs_dim`、`visual_encoder` 块），其余超参逐字不变，保证是 controlled comparison。生成后可以 `diff` 一下自查。

⚠️ **两个 suite 的 `experiment_name` 命名约定和 `obs_dim` 注释都不一样**，所以这两个锚点在脚本里是按正则匹配的，不是字面匹配。改动 config 模板时别把它们写死。

### 6.1 两个 suite 各 5 个 RGB checkpoint（2026-08-11 新增范围）

目标是拿到 **10 个 RL-finetuned RGB checkpoint**：`libero_object [1,6,7,8,9]` + `libero_spatial [0,2,3,6,7]`。这一版范围比原方案（只做 object）大，spatial 不再是「时间富裕才做」的可选项。

⚠️ 开跑前必须知道的两件事：

1. **仓库里原本只有 `libero_spatial_task0` 和 `task3` 两个 privileged config**，正文任务集里的 2/6/7 根本不存在，`make_rgb_configs.py` 会因为找不到源文件而失败。
2. **仅有的这两个还互相不一致**：task0 是 `micro_batch_size: 128`，task3 是 64；task3 还少了 `max_trials_per_task`。

处理方式：**5 个 spatial 的 privileged config 全部从 task0 模板重写成一致的版本**，只有 `specific_reset_id` 和 `experiment_name` 不同，并统一 `micro_batch_size: 64`（与 libero_object 全套一致）、`total_num_envs: 32`。理由是这 5 个 residual 之后要互相算 RFC 并 merge，跨任务超参必须一致才有可比性。

`micro_batch_size` 只影响梯度累积粒度，`global_batch_size` 两边都是 1024，数学上等价；但它会改变显存占用，换机器时注意。

⚠️ `total_num_envs` **统一用 32**。`libero_object_task1` 是全场唯一用 64 的（privileged 原版和 RGB 版都是，属于论文自带的不对称），而它恰好是收敛最差的一个 run，所以新增的 spatial 实验不沿用这个特例。

| suite | 任务 | 源 config | GPU |
|---|---|---|---|
| libero_object | 1, 6, 7, 8, 9 | 仓库原有（task1 的 `total_num_envs=64` 逐字保留） | 每 run 1 卡（task1 用 2 卡，因为 64 env） |
| libero_spatial | 0, 2, 3, 6, 7 | 由 task0 模板统一重写 | 每 run 1 卡 |

开关位置：
- `env.{train,eval}.obs_mode: rgb`（默认 `privileged`，不写就是老行为）
- `visual_encoder.{enabled,model_path,image_size,pooling}`
- `actor.model.obs_dim: 846`

⚠️ **两种模式的 checkpoint 不能互换**：`fc1_A` 的输入宽度不同（846 vs 88/74），加载会直接报形状错。

---

## 7. 一定要知道的坑

### 7.1 observation 维度是 per-suite 的

| suite | obs_dim | 拆解 |
|---|---|---|
| libero_object | 88 | 39 proprio + 49 物体关系（7 物体 × 7） |
| libero_spatial | 74 | 39 proprio + 35 物体关系（5 物体 × 7） |
| RGB 模式（任意 suite） | 846 | 2×384 视觉 + 2×39 proprio |

privileged 模式下**跨 suite 的 residual 无法 merge**——不是效果差，是张量形状对不上。RGB 模式下宽度与场景物体数无关，才能跨 suite。

### 7.1b ⚠️ 并行跑多个训练必须一个 run 一个容器

在**同一个容器**里同时起多个训练，它们会共用一个 Ray cluster：RLinf 的调度器通过扫 `/proc` 里带 `--gcs-address` 的进程来发现已有的 GCS（`ray/_private/services.py:_find_address_from_flag`），同一个 PID namespace 下后启动的 run 会 attach 到先启动的那个 cluster 上。

后果有两个，都很贵：

1. **杀掉任意一个 run 会拆掉整个 cluster**，其余 run 全部经 `rlinf/scheduler/cluster/cluster.py:signal_handler` 一起死，日志里表现为 `requests.exceptions.JSONDecodeError` 或 `The actor is dead because it was killed by ray.kill`——看不出是被连坐的。
2. **`cluster.component_placement` 的限卡在共享 cluster 下不起作用**，实测过一个 run 的 rollout 从 12 s 恶化到 397 s（30×）。

正确做法是**一个 run 一个容器**，并用 `--gpus device=N` 在设备层面限卡（这样调度器探测到的就是真实卡数，`component_placement` 保持 config 原样的 `all` 即可）：

```bash
docker run -d --gpus '"device=2"' --shm-size 64g --name rlinf-t6 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/workspace/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $RLINF_DIR:/workspace/RLinf -v $HF_HOME:/workspace/hf \
  rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 sleep infinity
```

⚠️ 训练容器**不要**加 `--net=host`，否则多个容器的 Ray 端口会撞车。模型都在挂载的 `HF_HOME` 里，训练不需要外网，加 `HF_HUB_OFFLINE=1` 即可。

完整的启动脚本见 `rebuttal/launch_rgb_runs.sh`。

⚠️ 另外，`save_interval` 默认是 1000，跑到 2000 步只有两个 checkpoint。并行跑多个 run 时建议降到 500，否则中途挂掉可能一个 checkpoint 都没有（实测被这条坑掉过 4 个 run × 4.5 小时）。

⚠️ **`docker exec ... bash -lc` 不会激活 venv**。镜像里那行 `source /opt/venv/openvla/bin/activate` 写在 `/root/.bashrc` 顶部的交互式分支之后，非交互 shell 读不到，表现是 `run_embodiment.sh: line 39: python: command not found`。启动训练一律显式加 PATH：

```bash
docker exec -w /workspace/RLinf rlinf-t6 bash -c \
  'export PATH=/opt/venv/openvla-oft/bin:$PATH; bash examples/embodiment/run_embodiment.sh <config>'
```

### 7.1c ⚠️ 「某个 run 莫名慢 10 倍」几乎总是外部资源争抢

实测过三次：t1 的 rollout 449 s、t7 280 s、s0 286 s，而健康值是 12–30 s。三次的共同点都是**同机另一个用户在 8 张卡上各占了 ~47 GB 的任务**，与 suite、config、任务 id 全都无关（同一份 spatial config，别人退出前 286 s、退出后 13 s）。

判断方法：看 tensorboard 的 `time/time/generate_rollouts`，它占 `time/time/step` 的绝大部分就是 rollout 被拖慢，而不是训练侧的问题。**不要因此去改 config**——`total_num_envs`、`max_trials_per_task`、env 基文件的继承关系都不是原因。

⚠️ 长期资源饥饿之后 run 可能**静默死掉**：mujoco 的 EnvWorker 子进程崩溃 → `venv.py:recv` 抛 `EOFError` → 整个 Ray cluster `ray.kill`。日志里没有任何 mujoco 侧的报错，只有一串 traceback。这类失败不会自愈，并行跑多个 run 时必须有独立的存活监控，否则会白等几小时：

```bash
docker exec rlinf-<run> pgrep -cf train_embodied_agent   # 0 = 已死
grep -aE "EOFError|Exiting main process|CUDA out of memory" rebuttal/runlogs/<run>.log
```

### 7.2 评测协议：32 个 episode ≠ 32 个初始状态

config 里 `env.eval.max_trials_per_task: 1` + `use_fixed_reset_state_ids: True` ⇒ **每个任务只用一个固定初始状态**，32 个 env 跑的是同一个场景。SR 的方差只来自 base policy 的解码采样（`do_sample: True`、`temperature_eval: 0.6`），不来自初始状态多样性。

要覆盖多个初始状态就把 `max_trials_per_task` 调大（LIBERO 每任务有 50 个初始状态），`total_num_envs` 不用动，GPU 成本基本不变。

⚠️ `eval_single_task.sh` 的 hydra 命令是写死的（`eval_single_task.sh:172`），**不透传额外的 override**。要加参数就直接调底层脚本：

```bash
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export PYTHONPATH=/workspace/RLinf MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
python $EMBODIED_PATH/eval_embodied_agent.py \
    --config-path $EMBODIED_PATH/config/ \
    --config-name libero_object_task1_lora_residual_sac_openvlaoft \
    runner.eval_policy_path="<checkpoint 路径>" \
    runner.only_eval=True \
    runner.logger.log_path=/workspace/RLinf/results/manual_eval \
    env.eval.specific_reset_id=1 \
    env.eval.max_trials_per_task=10
```

顺带说明：`rebuttal/openvla测试数据/` 里那批 SFT baseline 日志的 per-task SR 是 0% 或 100%，因为那是 openvla 自带的 eval 脚本（确定性解码）＋单一初始状态；RLinf 侧的评测不会这样。解析那批日志要用 `rebuttal/parse_sft_eval_logs.py`——日志里的 `# successes: N (X%)` 是**全局累计**计数，直接按任务读会得到完全错误的表。

### 7.3 渲染后端

- **训练** `run_embodiment.sh` 用 `MUJOCO_GL=egl`
- **评测** `eval_embodiment.sh`、`eval_single_task.sh`、`eval_merged_policies.sh`，以及 correction field 分析、smoke test，用 `MUJOCO_GL=osmesa`

两个都能用。egl 快一些但对驱动/`NVIDIA_DRIVER_CAPABILITIES` 敏感（所以容器要带 `graphics`）；osmesa 是纯 CPU 软渲染，最稳。报 EGL 相关错误时先换 osmesa 试。

### 7.4 手动跑 python 脚本要自己设环境变量

`*.sh` 会设，直接 `python xxx.py` 不会：

```bash
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

不设 `PYTHONPATH` 的表现是 `ModuleNotFoundError: No module named 'rlinf'`。

### 7.5 config 里的绝对路径

`residual_policy.base_model_path`、`actor.base_model.model_path`、`runner.eval_policy_path`、`runner.resume_dir` 全是 `/workspace/...` 开头的绝对路径，且带 HF snapshot 的 commit hash。**换机器后逐个确认**，不存在的路径通常表现为加载时报错或静默用了默认 config。

### 7.6 `.gitignore` 忽略了 `*.md` 和 `*.txt`

仓库根的 `.gitignore` 第 9 / 18 行忽略了所有 `.md` 和 `.txt`（README 等是历史上强加进来的）。新写的文档和实验结果 txt **不会**被 `git add` 捕获，需要 `git add -f`。好处是 rebuttal 的内部材料不会误提交到公开仓库。

---

## 8. 故障排查速查

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: No module named 'rlinf'` | 没设 `PYTHONPATH=/workspace/RLinf` |
| `ModuleNotFoundError: No module named 'torch'` | 没切 venv，或用了系统 `python3`；`source switch_env openvla-oft` |
| `OSError: We couldn't connect to 'https://huggingface.co'` | 容器没配代理，见 §2.1；或模型没进缓存 |
| 加载 checkpoint 报 shape mismatch（`fc1_A`） | obs_mode / suite 对不上，见 §7.1 |
| EGL / mujoco 渲染错误 | 换 `MUJOCO_GL=osmesa`；确认容器带 `NVIDIA_DRIVER_CAPABILITIES=...,graphics` |
| CUDA OOM | 减 `env.train.total_num_envs`、开 `enable_offload`、用 `CUDA_VISIBLE_DEVICES` 挑空卡 |
| 跨任务评测结果里缺了想要的任务 | `batch_eval_cross_task.sh:76` 的 `SKIP_TASKS` 硬编码，见 §5.3 |
| `global_batch_size must be divisible by ...` | 改了卡数没同步改 batch，见 §5.1 |
| `no viable alternative at input '{"actor,env,rollout"'` | placement 的 key 带逗号，不能命令行 override，见 §5.1 |
| 限制 GPU 数没生效 | `CUDA_VISIBLE_DEVICES` 对本调度器无效，必须改 config placement，见 §5.1 |
| RFC 输出全是 `CANNOT MERGE` | 硬编码阈值 0.7 太高，见 §5.4，看 `Mean` 自己比 τ |
| `run_embodiment.sh: line 39: python: command not found` | `docker exec bash -lc` 不激活 venv，要显式 `export PATH=/opt/venv/openvla-oft/bin:$PATH`，见 §7.1b |
| 某个 run 的 `generate_rollouts` 比别的慢 10 倍 | 同机他人占卡，不是 config 问题，见 §7.1c |
| 训练日志停在一串 traceback、`EOFError` + `ray.kill` | mujoco EnvWorker 子进程静默崩溃（常见于长期资源饥饿后），见 §7.1c；必须靠监控发现 |

---

## 9. 相关文档

- `README.md` — 代码结构与训练 pipeline 的模块说明
- `toolkits/merge_lora_policies/README.md` — merge 算法细节
- `examples/embodiment/eval/correction_field_analysis/DETAILED_GUIDE.md` — RFC 分析的详细说明
- `rebuttal/proposal-sim-openvlaoft.md` — 当前 rebuttal 实验的方案、已核实的事实与待决策项（不在 git 里）
