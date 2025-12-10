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
import torch.nn as nn
from torch.distributions.normal import Normal

from .modules.utils import layer_init


class ResidualActor(nn.Module):
    """
    Residual actor network for SAC.
    Outputs residual actions that are added to base actions.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_action_chunks: int = 1,
        obs_use_base_action: bool = False,
    ):
        """
        Initialize residual actor.
        
        Args:
            obs_dim: Dimension of observations
            action_dim: Dimension of actions
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_action_chunks: Number of action chunks (usually 1 for residual)
        """
        super().__init__()
        self.obs_use_base_action = obs_use_base_action
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        
        # Build network layers
        layers = []
        if obs_use_base_action:
            layers.append(layer_init(nn.Linear(self.obs_dim + action_dim, hidden_dim)))
        else:
            layers.append(layer_init(nn.Linear(self.obs_dim, hidden_dim)))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.ReLU())
        
        # Output: mean and log_std (2 * action_dim)
        layers.append(layer_init(nn.Linear(hidden_dim, action_dim * 2), std=0.01))
        self.network = nn.Sequential(*layers)
        
        # Action bounds (for tanh squashing)
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("action_bias", torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action mean and log_std.
        
        Args:
            obs: Observations, shape [batch_size, obs_dim]
            
        Returns:
            mean: Action mean, shape [batch_size, action_dim]
            log_std: Action log std, shape [batch_size, action_dim]
        """
        # Convert obs to match network dtype (e.g., bf16 to match base_model)
        network_dtype = (next(self.parameters()).dtype)
        obs = obs.to(dtype=network_dtype) # [batch_size, obs_dim]
        x = self.network(obs)
        mean, log_std = x.chunk(2, dim=-1)
        
        # Clamp log_std to reasonable range
        log_std = torch.tanh(log_std)
        log_std = -20 + 0.5 * (2 - (-20)) * (log_std + 1)  # [-20, 2]
        
        return mean, log_std
    
    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.
        
        Args:
            obs: Observations, shape [batch_size, obs_dim]
            
        Returns:
            action: Sampled actions, shape [batch_size, action_dim]
            log_prob: Log probabilities, shape [batch_size, 1]
            mean: Action means, shape [batch_size, action_dim]
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=False)  # Return [batch_size] instead of [batch_size, 1]
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action
    
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic action for evaluation.
        
        Args:
            obs: Observations, shape [batch_size, obs_dim]
            
        Returns:
            action: Deterministic actions, shape [batch_size, action_dim]
        """
        mean, log_std = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action
    
    def preprocess_obs(self, env_obs: dict) -> torch.Tensor:
        """
        Extract and preprocess observations from environment output.
        
        Args:
            env_obs: Environment observation dict
            
        Returns:
            obs: Processed observations, shape [batch_size, obs_dim]
        """
        obs_np = np.concatenate([env_obs["robot_proprio_state"], env_obs["object_to_robot_relations"]], -1)
       
        # Convert to tensor
        obs = torch.from_numpy(obs_np).float()
        
        return obs
    
    # def forward_training(
    #     self,
    #     data: dict,
    #     compute_logprobs: bool = True,
    #     compute_entropy: bool = False,
    #     compute_values: bool = False,
    #     **kwargs,
    # ) -> dict:
    #     """
    #     Forward pass for training (compatible with RLinf interface).
        
    #     Args:
    #         data: Dict containing 'obs' and 'action'
    #         compute_logprobs: Whether to compute log probabilities
    #         compute_entropy: Whether to compute entropy
    #         compute_values: Not used (residual actor doesn't have value head)
            
    #     Returns:
    #         Dict with logprobs and optionally entropy
    #     """
    #     obs = data["obs"]
    #     action = data["action"]
        
    #     # Call the tensor-based forward method
    #     mean, log_std = self.forward(obs)
    #     std = log_std.exp()
    #     normal = torch.distributions.Normal(mean, std)
        
    #     ret_dict = {}
    #     if compute_logprobs:
    #         # Compute log prob with tanh correction
    #         x_t = torch.atanh(torch.clamp((action - self.action_bias) / self.action_scale, -0.999, 0.999))
    #         log_prob = normal.log_prob(x_t)
    #         log_prob -= torch.log(self.action_scale * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
    #         ret_dict["logprobs"] = log_prob
        
    #     if compute_entropy:
    #         entropy = normal.entropy()
    #         ret_dict["entropy"] = entropy
        
    #     return ret_dict
    
    def predict_action_batch(
        self,
        env_obs, 
        base_actions: np.ndarray = None,
        calulate_logprobs: bool = True,
        calulate_values: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """
        Predict action batch (compatible with RLinf interface).
        
        Args:
            env_obs: Environment observations dict
            calulate_logprobs: Whether to compute log probabilities
            calulate_values: Not used (residual actor doesn't have value head)
            base_actions: Base actions, shape [batch_size, num_action_chunks, action_dim] 
            
        Returns:
            chunk_actions: numpy array [batch_size, num_action_chunks, action_dim]
            result: dict with prev_logprobs, prev_values, forward_inputs
        """
        obs = self.preprocess_obs(env_obs)
        device = next(self.parameters()).device
        obs = obs.to(device) # [batch_size, obs_dim]
        
        batch_size = obs.shape[0]
        
        if self.obs_use_base_action:
            base_actions_tensor = torch.from_numpy(base_actions).to(device).float()
            
            # # Reshape base_actions for batch processing: [batch_size * num_action_chunks, action_dim]
            base_actions_flat = base_actions_tensor.view(-1, self.action_dim) # [batch_size * num_action_chunks, action_dim]
            
            # # Repeat obs for each chunk: [batch_size * num_action_chunks, obs_dim]
            obs_repeated = obs.repeat_interleave(self.num_action_chunks, dim=0)
            
            # # Concatenate obs with base action: [batch_size * num_action_chunks, obs_dim + action_dim]
            obs_with_base_flat = torch.cat([obs_repeated, base_actions_flat], dim=-1)
            
            # Get residual actions for all chunks in one batch operation
            if calulate_logprobs:
                actions_flat, log_probs_flat, _ = self.get_action(obs_with_base_flat)
            else:
                actions_flat = self.get_eval_action(obs_with_base_flat)
                log_probs_flat = torch.zeros(batch_size * self.num_action_chunks, 1, device=device)
            
            # Reshape back to chunk format: [batch_size, num_action_chunks, action_dim]
            actions = actions_flat.view(batch_size, self.num_action_chunks, self.action_dim)
            log_probs = log_probs_flat.view(batch_size, self.num_action_chunks, 1)
            chunk_obs_with_base = obs_with_base_flat.view(batch_size, self.num_action_chunks, self.obs_dim + self.action_dim)
        
        else:
            if calulate_logprobs:
                actions_single, log_probs_single, _ = self.get_action(obs)
            else:
                actions_single = self.get_eval_action(obs)
                log_probs_single = torch.zeros(batch_size, 1, device=device)
            
            # Reshape to match expected format: repeat single action for all chunks
            actions = actions_single.unsqueeze(1).repeat(1, self.num_action_chunks, 1)
            log_probs = log_probs_single.unsqueeze(1).repeat(1, self.num_action_chunks, 1)
        
        # Convert to numpy
        # NOTE: Convert to float32 before numpy conversion (numpy doesn't support bf16)
        actions_float = actions.float() if actions.dtype == torch.bfloat16 else actions
        chunk_actions = actions_float.cpu().numpy()
        log_probs_float = log_probs.float() if log_probs.dtype == torch.bfloat16 else log_probs
        
        result = {
            "prev_logprobs": log_probs_float,
            "prev_values": None,  # Residual actor doesn't have value head
            "forward_inputs": {"obs": obs},  # NOTE: store the obs that was used to query actions
        }
        
        return chunk_actions, result

