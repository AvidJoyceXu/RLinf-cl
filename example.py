# libero/sac/residual_sac_libero.py
import os
import warnings
import random
import time
import yaml
from dataclasses import dataclass
import sys
# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 警告
# 设置HuggingFace缓存目录 - 使用环境变量或默认值
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
# 重定向stderr来过滤版本弃用警告
class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, message):
        if any(keyword in message for keyword in [
            "DeprecationWarning", 
            "distutils Version classes are deprecated",
            "LooseVersion",
            "packaging.version",
            "torch.meshgrid",
        ]):
            return
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
original_stderr = sys.stderr
sys.stderr = WarningFilter(original_stderr)
# 全面过滤警告
warnings.filterwarnings("ignore")

sys.path.append("/home/admin01/clproject/ContinualRL-LIBERO")
# # 设置代理环境变量
# PROXY_HOST = "127.0.0.1"
# PROXY_PORT = "7897"

# print("设置代理环境变量...")
# os.environ["http_proxy"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
# os.environ["https_proxy"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
# os.environ["HTTP_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
# os.environ["HTTPS_PROXY"] = f"http://{PROXY_HOST}:{PROXY_PORT}"
# os.environ["no_proxy"] = "localhost,127.0.0.1,::1,*.local"

# 抑制特定库的警告
try:
    import logging
    logging.getLogger("robosuite").setLevel(logging.ERROR)
    logging.getLogger("thop").setLevel(logging.ERROR)
    logging.getLogger("pkg_resources").setLevel(logging.ERROR)
    # 不抑制 wandb 警告
except:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# 导入 wandb
import wandb

from libero.sac.utils.replay_buffer import ReplayBuffer
from libero.sac.envs.libero_env import LiberoRLEnv

# 导入网络定义
from libero.sac.base_sac_libero import flatten_observation

# 导入BCTransformerPolicy
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.utils import get_task_embs
from libero.libero.benchmark import get_benchmark_dict

# 导入utils.profiling
from libero.sac.utils.profiling import NonOverlappingTimeProfiler
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个GPU

# 在文件开头添加全局缓存
_task_emb_cache = {}

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def load_base_model(base_model_path, device):
    """加载预训练的BC Policy"""
    print(f"Loading base model from: {base_model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(base_model_path, map_location=device)
    
    # 直接提取配置和状态字典
    cfg = checkpoint['cfg']  # 直接获取配置
    state_dict = checkpoint['state_dict']  # 直接获取状态字典
    
    # 创建shape_meta
    shape_meta = {
        "all_shapes": {
            "agentview_rgb": [3, 128, 128],
            "eye_in_hand_rgb": [3, 128, 128],
            "gripper_states": [2],
            "joint_states": [7]
        },
        "ac_dim": 7,
        "all_obs_keys": ["agentview_rgb", "eye_in_hand_rgb", "gripper_states", "joint_states"],
        "use_images": True
    }
    
    # 根据策略类型创建正确的模型
    policy_type = cfg.policy.policy_type
    print(f"Detected policy type: {policy_type}")
    
    if policy_type == "BCViLTPolicy":
        from libero.lifelong.models.bc_vilt_policy import BCViLTPolicy
        model = BCViLTPolicy(cfg, shape_meta).to(device)
        print("Using BCViLTPolicy for ViLT model")
    elif policy_type == "BCTransformerPolicy":
        from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
        model = BCTransformerPolicy(cfg, shape_meta).to(device)
        print("Using BCTransformerPolicy for Transformer model")
    elif policy_type == "BCRNNPolicy":
        from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy
        model = BCRNNPolicy(cfg, shape_meta).to(device)
        print("Using BCRNNPolicy for RNN model")
    else:
        # 默认使用BCTransformerPolicy
        from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
        model = BCTransformerPolicy(cfg, shape_meta).to(device)
        print(f"Unknown policy type {policy_type}, using BCTransformerPolicy as default")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("Base model loaded successfully!")
    return model, cfg

# def load_base_model(base_model_path, device):
#     """加载预训练的BC Transformer Policy"""
#     print(f"Loading base model from: {base_model_path}")
    
#     # 加载checkpoint
#     checkpoint = torch.load(base_model_path, map_location=device)
    
#     # 直接提取配置和状态字典
#     cfg = checkpoint['cfg']  # 直接获取配置
#     state_dict = checkpoint['state_dict']  # 直接获取状态字典
    
#     # 创建shape_meta
#     shape_meta = {
#         "all_shapes": {
#             "agentview_rgb": [3, 128, 128],
#             "eye_in_hand_rgb": [3, 128, 128],
#             "gripper_states": [2],
#             "joint_states": [7]
#         },
#         "ac_dim": 7,
#         "all_obs_keys": ["agentview_rgb", "eye_in_hand_rgb", "gripper_states", "joint_states"],
#         "use_images": True
#     }
    
#     # 创建模型
#     model = BCTransformerPolicy(cfg, shape_meta).to(device)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
    
#     print("Base model loaded successfully!")
#     return model, cfg

def process_obs_for_base_model(obs, base_cfg, task_suite_name, task_id, device):
    """将环境观察处理为base model需要的data形式"""
    # 提取需要的观察模态
    joint_states = torch.tensor(obs['joint_states'], dtype=torch.float32, device=device)  # [num_envs, 7]
    gripper_states = torch.tensor(obs['gripper_states'], dtype=torch.float32, device=device)  # [num_envs, 2]
    agentview_rgb = torch.tensor(obs['agentview_rgb'], dtype=torch.float32, device=device)  # [num_envs, 3, 128, 128]
    eye_in_hand_rgb = torch.tensor(obs['eye_in_hand_rgb'], dtype=torch.float32, device=device)  # [num_envs, 3, 128, 128]
    
    # 获取任务嵌入 - 使用缓存
    cache_key = f"{task_suite_name}_{task_id}"
    
    if cache_key not in _task_emb_cache:
        # 实例化benchmark类
        benchmark_class = get_benchmark_dict()[task_suite_name]
        benchmark = benchmark_class()
        
        # 获取任务描述
        task = benchmark.get_task(task_id)
        task_description = task.language
        
        # 获取任务嵌入
        task_embs = get_task_embs(base_cfg, [task_description])
        task_emb = task_embs[0]  # 取第一个任务的嵌入
        
        # 缓存任务嵌入
        _task_emb_cache[cache_key] = task_emb
    
    # 使用缓存的任务嵌入
    task_emb = _task_emb_cache[cache_key]
    
    # 关键：task_emb 需要扩展到 [num_envs, 768] 格式
    num_envs = joint_states.shape[0]
    
    # 处理 task_emb 的维度 - 使用 repeat() 创建独立副本
    task_emb = task_emb.unsqueeze(0).repeat(num_envs, 1)  # [num_envs, 768]
    
    # 构建base model需要的data格式
    data = {
        'obs': {
            'joint_states': joint_states,      # [num_envs, 7]
            'gripper_states': gripper_states,  # [num_envs, 2]
            'agentview_rgb': agentview_rgb,    # [num_envs, 3, 128, 128]
            'eye_in_hand_rgb': eye_in_hand_rgb # [num_envs, 3, 128, 128]
        },
        'task_emb': task_emb.to(device),      # [num_envs, 768]
        'actions': torch.zeros(1, 7, device=device)  # 占位符
    }
    
    return data

def make_env(config, seed, idx, capture_video, log_path=None):
    """创建LIBERO环境的工厂函数"""
    def thunk():
        # 创建配置对象
        class Config:
            def __init__(self):
                self.seed = seed
                self.task_suite_name = config['environment']['task_suite_name']
                self.num_tasks_per_suite = config['environment']['num_tasks_per_suite']
                self.n_rollout_threads = config['environment']['n_rollout_threads']
                self.num_trials_per_task = config['environment']['num_trials_per_task']
                self.max_env_length = config['environment']['max_env_length']
                self.num_steps_wait = config['environment']['num_steps_wait']
                self.env_gpu_id = config['environment']['env_gpu_id']
                self.save_video = config['environment']['save_video'] and idx == 0
                
                self.model_family = "openvla"
                
                # 修改：使用传入的log_path
                if log_path is not None:
                    self.exp_dir = os.path.join(log_path, "rollouts")
                else:
                    # 备用方案：使用原来的逻辑
                    log_dir = config['misc']['log_dir']
                    self.exp_dir = os.path.join(log_dir, "rollouts")

                # 新增：单任务模式参数
                self.single_env_id = config['environment']['single_env_id']
                self.use_single_env = config['environment']['use_single_env']
        
        cfg = Config()
        
        # 直接使用 LiberoRLEnv
        env = LiberoRLEnv(cfg, mode="train")
        
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    """Q网络，只处理LIBERO的RL观察空间"""
    def __init__(self, env):
        super().__init__()
        
        # 计算观察空间的总维度
        obs_dim = self._get_observation_dim(env)
        action_dim = np.prod(env.action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def _get_observation_dim(self, env):
        """计算LIBERO RL观察空间的总维度"""
        total_dim = 0
        
        # 只使用RL部分
        # 完整机器人状态（RL使用）
        total_dim += 39  # robot0_proprio-state 的实际维度
        
        # 物体与机器人的关系（RL使用）
        total_dim += 35  # object_to_robot_relations 的实际维度
        
        return total_dim

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ResidualActor(nn.Module):
    """Residual Actor网络"""
    
    def __init__(self, env, config):
        super().__init__()
        
        # 计算观察空间的总维度
        obs_dim = self._get_observation_dim(env)
        
        # 修复：使用标准的gym space获取动作维度
        action_dim = np.prod(env.action_space.shape)  # 现在env.action_space是gym.spaces.Box
        
        # 根据actor_input决定输入维度
        if config['network']['actor_input'] == "obs":
            input_dim = obs_dim
        else:  # obs_base_action
            input_dim = obs_dim + action_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, action_dim), std=0.01)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(1.0, dtype=torch.float32),  # residual动作通常不需要缩放
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(0.0, dtype=torch.float32),
        )

    def _get_observation_dim(self, env):
        """计算LIBERO RL观察空间的总维度"""
        total_dim = 0
        
        # 只使用RL部分
        total_dim += 39  # robot0_proprio-state
        total_dim += 35  # object_to_robot_relations
        
        return total_dim
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = -20 + 0.5 * (2 - (-20)) * (log_std + 1)
        # LOG_STD_MIN=-20
        # LOG_STD_MAX=2
        return mean, log_std
    
    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def get_eval_action(self, x):
        mean, log_std = self(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

def evaluate(n, base_model, residual_actor, eval_env, device, config, base_cfg):
    print('======= Evaluation Starts =========')
    residual_actor.eval()
    result = defaultdict(list)
    
    obs, info = eval_env.reset()
    episode_count = 0
    episode_return = 0
    episode_length = 0
    
    while episode_count < n:
        # 处理观察
        obs_for_rl = flatten_observation(obs)
        obs_for_rl = torch.tensor(obs_for_rl, device=device)  # 移到GPU
        
        # 获取base action
        data_for_base = process_obs_for_base_model(
            obs, base_cfg, config['environment']['task_suite_name'], config['environment']['single_env_id'], device
        )
        with torch.no_grad():
            base_actions = base_model.get_action(data_for_base)
        
        # 获取residual action
        with torch.no_grad():
            actor_input = torch.Tensor(obs_for_rl).to(device) if config['network']['actor_input'] == 'obs' else torch.cat([torch.Tensor(obs_for_rl).to(device), torch.Tensor(base_actions).to(device)], dim=1)
            res_actions = residual_actor.get_eval_action(actor_input).detach().cpu().numpy()
        
        # 计算最终动作
        res_scale = config['training']['res_scale']
        scaled_res_actions = res_scale * res_actions
        final_actions = base_actions + scaled_res_actions
        
        # 执行环境步骤
        obs, rewards, dones, info = eval_env.step(final_actions)
        
        # 累计episode信息
        episode_return += rewards[0]  # 假设只评估第一个环境
        episode_length += 1
        
        # 检查episode是否结束
        if dones[0]:  # 第一个环境结束
            # 计算成功率：奖励>0表示成功
            success = 1.0 if episode_return > 0 else 0.0
            
            print(f"eval: ep_return={episode_return:.2f}, ep_len={episode_length}, success={success}")
            
            # 保存结果
            result['return'].append(episode_return)
            result['len'].append(episode_length)
            result['success'].append(success)
            
            # 重置episode统计
            episode_count += 1
            episode_return = 0
            episode_length = 0
            
            # 如果还需要更多episode，重置环境
            if episode_count < n:
                obs, info = eval_env.reset()
    
    print('======= Evaluation Ends =========')
    residual_actor.train()
    return result

def preload_task_embeddings(config, base_cfg, device):
    """预加载所有任务的任务嵌入"""
    print("Preloading task embeddings...")
    start_time = time.time()
    
    task_suite_name = config['environment']['task_suite_name']
    benchmark_class = get_benchmark_dict()[task_suite_name]
    benchmark = benchmark_class()
    
    for task_id in range(benchmark.n_tasks):
        task = benchmark.get_task(task_id)
        task_description = task.language
        
        # 获取任务嵌入
        task_embs = get_task_embs(base_cfg, [task_description])
        task_emb = task_embs[0]
        
        # 缓存任务嵌入
        cache_key = f"{task_suite_name}_{task_id}"
        _task_emb_cache[cache_key] = task_emb
        
        print(f"Preloaded task {task_id}: {task_description[:50]}...")
    
    print(f"Preloaded {benchmark.n_tasks} task embeddings in {time.time() - start_time:.2f}s")

def main():
    # 加载配置文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configs", "residual_sac_config.yaml")
    config = load_config(config_path)
    
    # 处理相对路径：如果base_model路径不是绝对路径，则相对于script_dir
    base_model_path = config['base_model']['path']
    if not os.path.isabs(base_model_path):
        base_model_path = os.path.join(script_dir, base_model_path)
    config['base_model']['path'] = base_model_path
    
    # 处理log_dir路径：如果不是绝对路径，则相对于script_dir
    log_dir = config['misc']['log_dir']
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(script_dir, log_dir)
    config['misc']['log_dir'] = log_dir
    
    # 从配置文件读取GPU设置（如果有的话）
    if 'gpu_id' in config.get('misc', {}):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['misc']['gpu_id'])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 默认使用第0张卡

    
    print(f"Using base_model_path: {config['base_model']['path']}")
    print(f"Single task mode: {config['environment']['use_single_env']}, Task ID: {config['environment']['single_env_id']}")
    
    # 验证base_model_path
    if not os.path.exists(config['base_model']['path']):
        print(f"Error: Base model path '{config['base_model']['path']}' does not exist!")
        return
    
    # 创建运行名称 - 使用详细的时间戳，并加上task id信息
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取task id信息
    task_id = config['environment']['single_env_id']
    
    # 修改run_name格式，加入task id
    run_name = f"{config['environment']['task_suite_name']}__task{task_id}__{config['misc']['exp_name']}__{config['misc']['seed']}__{timestamp}"
    
    # 从配置中读取log_dir
    log_dir = config['misc']['log_dir']
    log_path = os.path.join(log_dir, run_name)
    
    # 创建日志目录
    os.makedirs(log_path, exist_ok=True)
    
    # 初始化 wandb
    if config['wandb']['enabled']:
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=run_name,
            config=config,
            tags=["residual_sac", "libero", "base+residual", f"task{task_id}"]
        )
        print(f"Wandb initialized: {config['wandb']['entity']}/{config['wandb']['project']}/{run_name}")
    
    # 设置tensorboard - 使用从配置读取的log_path
    writer = SummaryWriter(log_path)
    
    # 设置随机种子
    random.seed(config['misc']['seed'])
    np.random.seed(config['misc']['seed'])
    torch.manual_seed(config['misc']['seed'])
    torch.backends.cudnn.deterministic = config['misc']['torch_deterministic']
    
    device = torch.device("cuda" if torch.cuda.is_available() and config['misc']['cuda'] else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    # 创建环境 - 传入log_path
    env = make_env(config, config['misc']['seed'], 0, config['misc']['capture_video'], log_path)()
    
    # 打印并行化环境信息
    env.print_environment_info()
    
    # 加载base model
    base_model, base_cfg = load_base_model(config['base_model']['path'], device)
    
    # 预加载任务嵌入
    preload_task_embeddings(config, base_cfg, device)
    
    # 创建residual actor和critic
    residual_actor = ResidualActor(env, config).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    
    # 复制参数到目标网络
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # 创建优化器
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config['training']['q_lr'])
    actor_optimizer = optim.Adam(list(residual_actor.parameters()), lr=config['training']['policy_lr'])
    
    # 检查模型是否在GPU上
    print(f"Base model device: {next(base_model.parameters()).device}")
    print(f"Residual actor device: {next(residual_actor.parameters()).device}")
    print(f"Qf1 device: {next(qf1.parameters()).device}")
    print(f"Qf2 device: {next(qf2.parameters()).device}")
    
    # 自动熵调优
    if config['training']['autotune']:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=config['training']['q_lr'])
    else:
        alpha = config['training']['alpha']
    
    # 根据critic_input和actor_input决定action space
    if config['network']['critic_input'] == 'res' and config['network']['actor_input'] == 'obs':
        action_shape = (7,)  # 只保存residual动作
    else:
        action_shape = (21,)  # residual(7) + base(7) + base_next(7)
    
    # 创建经验回放缓冲区
    obs_shape = (74,)  # 39 + 35 = 74
    class SimpleObsSpace:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = np.float32
    
    class SimpleActionSpace:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = np.float32
    
    simple_obs_space = SimpleObsSpace(obs_shape)
    simple_action_space = SimpleActionSpace(action_shape)
    
    rb = ReplayBuffer(
        config['training']['buffer_size'],
        simple_obs_space,
        simple_action_space,
        device,
        n_envs=config['environment']['n_rollout_threads'],
        handle_timeout_termination=False,
    )
    
    start_time = time.time()
    
    # 开始训练
    obs, _ = env.reset()
    obs_for_rl = flatten_observation(obs)  # 用于RL的扁平化观察
    print("obs_for_rl shape: ", obs_for_rl.shape)
    
    global_step = 0
    global_update = 0
    learning_has_started = False
    num_updates_per_training = int(config['training']['training_freq'] * config['training']['utd'])
    
    # 创建evaluation环境 - 也传入log_path
    eval_env = make_env(config, config['misc']['seed'] + 1000, 0, config['misc']['capture_video'], log_path)()
    eval_env.reset()

    # 初始化result字典
    result = defaultdict(list)
    timer = NonOverlappingTimeProfiler()

    # 初始化episode统计
    episode_buffer = []  # 存储所有episode的结果
    episode_count = 0   # 总episode计数
    report_interval = 5  # 每10个episode报告一次

    # 训练循环
    while global_step < config['training']['total_timesteps']:
        # Collect samples from environments
        for local_step in range(config['training']['training_freq'] // config['environment']['n_rollout_threads']):
            global_step += 1 * config['environment']['n_rollout_threads']

            # 1. 将obs处理为base model需要的data形式
            data_for_base = process_obs_for_base_model(
                obs, base_cfg, config['environment']['task_suite_name'], config['environment']['single_env_id'], device
            )
            
            # 2. 获取base model的基础动作
            with torch.no_grad():
                base_actions = base_model.get_action(data_for_base)  # [num_envs, 7]
            
            # 3. 生成residual动作
            res_ratio = min(global_step / config['training']['prog_explore'], 1)
            enable_res_masks = np.random.rand(config['environment']['n_rollout_threads']) < res_ratio

            # 修复：确保enable_res_masks是数组
            if global_step <= config['training']['prog_explore_threshold']:
                enable_res_masks = np.zeros(config['environment']['n_rollout_threads'], dtype=bool)  # 全部设为False

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                res_actions = np.zeros_like(base_actions, dtype=np.float32)
                # 学习未开始时，res_norm_ratio设为0
                res_norm_ratio_enabled = 0.0
                res_norm_ratio_all = 0.0
            else:
                # 生成residual动作
                actor_input = torch.Tensor(obs_for_rl).to(device) if config['network']['actor_input'] == 'obs' else torch.cat([torch.Tensor(obs_for_rl).to(device), torch.Tensor(base_actions).to(device)], dim=1)
                
                res_actions, _, _ = residual_actor.get_action(actor_input)
                res_actions = res_actions.detach().cpu().numpy()
                
                # 计算res_norm_ratio - 两种方式
                res_norm = np.linalg.norm(res_actions, axis=1)  # residual动作的L2范数
                base_norm = np.linalg.norm(base_actions, axis=1)  # base动作的L2范数
                
                # 1. res_norm_ratio_enabled: 只考虑被启用的环境的residual动作
                if np.any(enable_res_masks):
                    enabled_res_norms = res_norm[enable_res_masks]
                    enabled_base_norms = base_norm[enable_res_masks]
                    res_norm_ratio_enabled = np.mean(enabled_res_norms / (enabled_base_norms + 1e-8))
                else:
                    # 如果没有环境被启用，设为0
                    res_norm_ratio_enabled = 0.0
                
                # 2. res_norm_ratio_all: 考虑所有环境的residual动作（包括被mask的）
                res_norm_ratio_all = np.mean(res_norm / (base_norm + 1e-8))
                
                # 应用masking
                res_actions[~enable_res_masks] = 0.0

            # 4. 最终动作 = base + scaled_residual
            res_scale = config['training']['res_scale']
            scaled_res_actions = res_scale * res_actions
            final_actions = base_actions + scaled_res_actions
            if global_step % 1000 == 0:
                print(f"global_step: {global_step}")
                print(f"final_actions: {final_actions}")
                print(f"base_actions: {base_actions}")
                print(f"scaled_res_actions: {scaled_res_actions}")
                print(f"enable_res_masks: {enable_res_masks}")
                print(f"res_norm_ratio_enabled: {res_norm_ratio_enabled:.4f}")
                print(f"res_norm_ratio_all: {res_norm_ratio_all:.4f}")
                print("--------------------------------")
            
            # 5. 执行环境步骤
            next_obs, rewards, dones, infos = env.step(final_actions)
            next_obs_for_rl = flatten_observation(next_obs)  # 用于RL

            # 6. 修改奖励：rewards = rewards - 1.0
            rewards = rewards - 1.0  # negative reward + bootstrap at truncated yields best results

            # 新增：处理最后一帧obs的逻辑
            real_next_obs = next_obs.copy()
            real_next_obs_for_rl = next_obs_for_rl.copy()

            # 根据bootstrap_at_done策略处理
            if config['training']['bootstrap_at_done'] == 'never':
                stop_bootstrap = dones  # always stop bootstrap when episode ends
            else:
                if config['training']['bootstrap_at_done'] == 'always':
                    need_final_obs = dones  # always need final obs when episode ends
                    stop_bootstrap = np.zeros_like(dones, dtype=bool)  # never stop bootstrap
                else:  # bootstrap at truncated
                    # 这里需要根据你的环境实现来判断truncated vs terminated
                    # 假设dones包含termination信息，truncations需要从infos中获取
                    has_truncations = 'truncations' in infos
                    if has_truncations:
                        truncations = infos['truncations']
                        # print(f"[DEBUG] Using truncations from infos: {truncations}")
                    else:
                        truncations = np.zeros_like(dones, dtype=bool)
                        # print("[DEBUG] infos has no 'truncations'; using zeros_like(dones)")
                    need_final_obs = truncations & (~dones)  # only need final obs when truncated and not terminated
                    stop_bootstrap = dones  # only stop bootstrap when terminated, don't stop when truncated
                
                # 处理需要final obs的情况
                for idx, _need_final_obs in enumerate(need_final_obs):
                    if _need_final_obs:
                        # 这里需要根据你的环境实现来获取final_observation
                        # 可能需要从infos中获取
                        if 'final_observation' in infos:
                            real_next_obs[idx] = infos["final_observation"][idx]
                            real_next_obs_for_rl[idx] = flatten_observation(infos["final_observation"][idx])
            
            # 7. 获取下一个状态的base动作（用于保存到RB）
            data_for_base_next = process_obs_for_base_model(
                real_next_obs, base_cfg, config['environment']['task_suite_name'], config['environment']['single_env_id'], device
            )
            with torch.no_grad():
                base_next_actions = base_model.get_action(data_for_base_next)  # [num_envs, 7]
            
            # 8. 决定保存到RB的action格式
            if config['network']['critic_input'] == 'res' and config['network']['actor_input'] == 'obs':
                actions_to_save = res_actions
            else:
                # 直接使用 numpy 数组，不需要 cpu().numpy()
                actions_to_save = np.concatenate([res_actions, base_actions, base_next_actions], axis=1)
            
            # 9. 保存到经验回放缓冲区
            rb.add(obs_for_rl, real_next_obs_for_rl, actions_to_save, rewards, stop_bootstrap, infos)

            # 修改：收集episode信息 - 累积统计方式
            for i, reward in enumerate(rewards):
                if stop_bootstrap[i]:  # episode结束（成功或达到最大长度）
                    episode_count += 1
                    success = 1.0 if reward > -1 else 0.0  # 奖励>-1表示成功
                    episode_return = reward
                    
                    # 添加到episode缓冲区
                    episode_buffer.append({
                        'return': episode_return,
                        'success': success,
                        'res_norm_ratio_enabled': res_norm_ratio_enabled,  # 只考虑启用的环境
                        'res_norm_ratio_all': res_norm_ratio_all,  # 考虑所有环境
                        'step': global_step
                    })
                    
                    print(f"Episode {episode_count} finished: return={episode_return:.2f}, success={success}, res_norm_ratio_enabled={res_norm_ratio_enabled:.4f}, res_norm_ratio_all={res_norm_ratio_all:.4f}")

            # 每达到report_interval个episode就报告一次统计
            if len(episode_buffer) >= report_interval:
                # 计算最近report_interval个episode的统计
                recent_episodes = episode_buffer[-report_interval:]
                
                avg_return = np.mean([ep['return'] for ep in recent_episodes])
                avg_success_rate = np.mean([ep['success'] for ep in recent_episodes])
                avg_res_norm_ratio_enabled = np.mean([ep['res_norm_ratio_enabled'] for ep in recent_episodes])  # 只考虑启用的环境
                avg_res_norm_ratio_all = np.mean([ep['res_norm_ratio_all'] for ep in recent_episodes])  # 考虑所有环境
                if global_step % 1000 == 0:
                    print(f"=== Episode Statistics (last {report_interval} episodes) ===")
                    print(f"Average Return: {avg_return:.2f}")
                    print(f"Success Rate: {avg_success_rate:.2f}")
                    print(f"Average Res Norm Ratio (Enabled): {avg_res_norm_ratio_enabled:.4f}")
                    print(f"Average Res Norm Ratio (All): {avg_res_norm_ratio_all:.4f}")
                    print(f"Total Episodes: {episode_count}")
                
                # 记录到tensorboard
                writer.add_scalar("train/episodic_return", avg_return, global_step)
                writer.add_scalar("train/success_rate", avg_success_rate, global_step)
                writer.add_scalar("train/res_norm_ratio_enabled", avg_res_norm_ratio_enabled, global_step)  # 只考虑启用的环境
                writer.add_scalar("train/res_norm_ratio_all", avg_res_norm_ratio_all, global_step)  # 考虑所有环境
                writer.add_scalar("train/total_episodes", episode_count, global_step)
                
                # 同时记录到 wandb
                if config['wandb']['enabled']:
                    wandb.log({
                        "train/episodic_return": avg_return,
                        "train/success_rate": avg_success_rate,
                        "train/res_norm_ratio_enabled": avg_res_norm_ratio_enabled,  # 只考虑启用的环境
                        "train/res_norm_ratio_all": avg_res_norm_ratio_all,  # 考虑所有环境
                        "train/total_episodes": episode_count,
                        "train/global_step": global_step  # 添加global step记录
                    }, step=global_step)

            timer.end('collect')
            # 更新观察
            obs = real_next_obs
            obs_for_rl = real_next_obs_for_rl
        
        # ALGO LOGIC: training.
        if rb.size() * rb.n_envs < config['training']['learning_starts']:
            continue

        learning_has_started = True
        for local_update in range(num_updates_per_training):
            global_update += 1
            
            data = rb.sample(config['training']['batch_size'])
            
            # 解析保存的actions
            if config['network']['critic_input'] != 'res' or config['network']['actor_input'] == 'obs_base_action':
                res_actions = data.actions[:, :7]  # 前7维是residual
                base_actions = data.actions[:, 7:14]  # 中间7维是当前base
                base_next_actions = data.actions[:, 14:]  # 后7维是下一个状态base
            else:
                res_actions = data.actions

            #############################################
            # Train agent
            #############################################
            
            # 训练Q网络
            with torch.no_grad():
                # 构建actor输入
                if config['network']['actor_input'] == 'obs':
                    actor_input = data.next_observations
                else:
                    actor_input = torch.cat([data.next_observations, base_next_actions], dim=1)
                
                # 获取下一个状态的residual动作
                next_state_res_actions, next_state_log_pi, _ = residual_actor.get_action(actor_input)
                
                # 根据critic_input模式构建下一个状态的动作
                if config['network']['critic_input'] == 'res':
                    next_state_actions = next_state_res_actions
                elif config['network']['critic_input'] == 'sum':
                    scaled_res_actions = config['training']['res_scale'] * next_state_res_actions
                    next_state_actions = base_next_actions + scaled_res_actions
                else:  # concat
                    next_state_actions = torch.cat([next_state_res_actions, base_next_actions], dim=1)
                
                # 计算Q值
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * config['training']['gamma'] * (min_qf_next_target).view(-1)
            
            # 根据critic_input模式构建当前状态的动作
            if config['network']['critic_input'] == 'res':
                current_actions = res_actions
            elif config['network']['critic_input'] == 'sum':
                scaled_res_actions = config['training']['res_scale'] * res_actions
                current_actions = base_actions + scaled_res_actions
            else:  # concat
                current_actions = torch.cat([res_actions, base_actions], dim=1)
            
            qf1_a_values = qf1(data.observations, current_actions).view(-1)
            qf2_a_values = qf2(data.observations, current_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            q_optimizer.zero_grad()
            qf_loss.backward()
            qf1_grad_norm = nn.utils.clip_grad_norm_(qf1.parameters(), config['training']['max_grad_norm'])
            qf2_grad_norm = nn.utils.clip_grad_norm_(qf2.parameters(), config['training']['max_grad_norm'])
            q_optimizer.step()
            
            # 训练策略网络
            if global_step % config['training']['policy_frequency'] == 0:
                for _ in range(config['training']['policy_frequency']):
                    # 构建actor输入
                    if config['network']['actor_input'] == 'obs':
                        actor_input = data.observations
                    else:
                        actor_input = torch.cat([data.observations, base_actions], dim=1)
                    
                    pi, log_pi, _ = residual_actor.get_action(actor_input)
                    
                    # 根据critic_input模式构建动作
                    if config['network']['critic_input'] == 'res':
                        pi_actions = pi
                    elif config['network']['critic_input'] == 'sum':
                        scaled_res_actions = config['training']['res_scale'] * pi
                        pi_actions = base_actions + scaled_res_actions
                    else:  # concat
                        pi_actions = torch.cat([pi, base_actions], dim=1)
                    
                    qf1_pi = qf1(data.observations, pi_actions)
                    qf2_pi = qf2(data.observations, pi_actions)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(residual_actor.parameters(), config['training']['max_grad_norm'])
                    actor_optimizer.step()
                    
                    if config['training']['autotune']:
                        with torch.no_grad():
                            _, log_pi, _ = residual_actor.get_action(actor_input)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
            
            # 更新目标网络
            if global_step % config['training']['target_network_frequency'] == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(config['training']['tau'] * param.data + (1 - config['training']['tau']) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(config['training']['tau'] * param.data + (1 - config['training']['tau']) * target_param.data)
            
            # 记录训练信息
            if (global_step - config['training']['training_freq']) // config['training']['log_freq'] < global_step // config['training']['log_freq']:
                if len(result['return']) > 0:
                    for k, v in result.items():
                        writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                        # 同时记录到wandb
                        if config['wandb']['enabled']:
                            wandb.log({f"train/{k}": np.mean(v), "train/global_step": global_step}, step=global_step)
                    result = defaultdict(list)
                
                # 记录训练损失
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/qf1_grad_norm", qf1_grad_norm.item(), global_step)
                writer.add_scalar("losses/qf2_grad_norm", qf2_grad_norm.item(), global_step)
                writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
                
                # 记录到 wandb
                if config['wandb']['enabled']:
                    wandb.log({
                        "losses/qf1_values": qf1_a_values.mean().item(),
                        "losses/qf2_values": qf2_a_values.mean().item(),
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/qf2_loss": qf2_loss.item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/actor_loss": actor_loss.item(),
                        "losses/alpha": alpha,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                        "train/global_step": global_step  # 添加global step记录
                    }, step=global_step)
                if global_step % 1000 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                
                if config['training']['autotune']:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                    if config['wandb']['enabled']:
                        wandb.log({"losses/alpha_loss": alpha_loss.item(), "train/global_step": global_step}, step=global_step)
        
        # # Evaluation
        # if (global_step - config['training']['training_freq']) // config['training']['eval_freq'] < global_step // config['training']['eval_freq']:
        #     print("=============== start eval =================")
            
        #     result = evaluate(config['training']['num_eval_episodes'], base_model, residual_actor, eval_env, device, config, base_cfg)
        #     for k, v in result.items():
        #         writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
        #         # 同时记录到wandb
        #         if config['wandb']['enabled']:
        #             wandb.log({f"eval/{k}": np.mean(v)}, step=global_step)
            
        #     # 额外记录评估的详细信息
        #     if config['wandb']['enabled']:
        #         wandb.log({
        #             "eval/avg_return": np.mean(result['return']),
        #             "eval/avg_success_rate": np.mean(result['success']),
        #             "eval/avg_episode_length": np.mean(result['len']),
        #             "eval/total_episodes": len(result['return'])
        #         }, step=global_step)
        
        # Checkpoint
        if config['training']['save_freq'] and (global_step >= config['training']['total_timesteps'] or \
                (global_step - config['training']['training_freq']) // config['training']['save_freq'] < global_step // config['training']['save_freq']):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'res_actor': residual_actor.state_dict(),
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
                'log_alpha': log_alpha if config['training']['autotune'] else np.log(config['training']['alpha']),
            }, f'{log_path}/checkpoints/{global_step}.pt')
            print(f"Saved checkpoint at step {global_step} to {log_path}/checkpoints/{global_step}.pt")
    
    # 保存最终模型
    print(f"Saving final model at step {global_step} to {log_path}")
    os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
    torch.save({
        'res_actor': residual_actor.state_dict(),
        'qf1': qf1_target.state_dict(),
        'qf2': qf2_target.state_dict(),
        'log_alpha': log_alpha if config['training']['autotune'] else np.log(config['training']['alpha']),
    }, f'{log_path}/checkpoints/final.pt')
    
    # 关闭环境
    env.close()
    eval_env.close()
    writer.close()
    
    # 关闭 wandb
    if config['wandb']['enabled']:
        wandb.finish()

if __name__ == "__main__":
    main() 