# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from typing import Union, Dict, Any

def flatten_libero_rl_observation(obs: Dict[str, Any]) -> np.ndarray:
    """
    Extract and flatten RL observations from LIBERO environment output.
    
    env_obs keys (from RLinf LiberoEnv._wrap_obs):
    - images
    - wrist_images: None
    - states: shape (num_group_envs, 8)
    states = np.concatenate([
                obs["robot0_eef_pos"], # 3D
                quat2axisangle(obs["robot0_eef_quat"]), # 3D
                obs["robot0_gripper_qpos"], # 2D
            ]),
    - robot_proprio_state: numpy array [num_envs, proprio_dim] (39D)
    - object_to_robot_relations: numpy array [num_envs, relation_dim] (35D)
    - task_descriptions: list of strings
    
    Returns:
        Flattened RL observations: [robot_proprio_state, object_to_robot_relations]
        Shape: [num_envs, proprio_dim + relation_dim] or [proprio_dim + relation_dim]
    """
    flattened = []
    
    # 只使用RL部分
    # 完整机器人状态 - 处理向量化数据
    if obs["robot_proprio_state"].ndim == 2:
        # 如果是 (env_num, dim) 格式，展平每个环境
        for i in range(obs["robot_proprio_state"].shape[0]):
            env_obs = []
            env_obs.append(obs["robot_proprio_state"][i].flatten())
            env_obs.append(obs["object_to_robot_relations"][i].flatten())
            flattened.append(np.concatenate(env_obs))
    else:
        # 如果是单个环境的观察
        flattened.append(obs["robot_proprio_state"].flatten())
        flattened.append(obs["object_to_robot_relations"].flatten())
        flattened = [np.concatenate(flattened)]
    
    return np.array(flattened)
