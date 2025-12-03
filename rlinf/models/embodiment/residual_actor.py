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
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        
        # Build network layers
        layers = []
        layers.append(layer_init(nn.Linear(obs_dim, hidden_dim)))
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
        network_dtype = next(self.network.parameters()).dtype
        obs = obs.to(dtype=network_dtype)
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
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
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
        # Use utility function for LIBERO to get proper RL observations
        from rlinf.envs.libero.rl_obs_utils import flatten_libero_rl_observation
        
        obs_np = flatten_libero_rl_observation(env_obs)
       
        # Convert to tensor
        obs = torch.from_numpy(obs_np).float()
        
        return obs
    
    def forward_training(
        self,
        data: dict,
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ) -> dict:
        """
        Forward pass for training (compatible with RLinf interface).
        
        Args:
            data: Dict containing 'obs' and 'action'
            compute_logprobs: Whether to compute log probabilities
            compute_entropy: Whether to compute entropy
            compute_values: Not used (residual actor doesn't have value head)
            
        Returns:
            Dict with logprobs and optionally entropy
        """
        obs = data["obs"]
        action = data["action"]
        
        # Call the tensor-based forward method
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        ret_dict = {}
        if compute_logprobs:
            # Compute log prob with tanh correction
            x_t = torch.atanh(torch.clamp((action - self.action_bias) / self.action_scale, -0.999, 0.999))
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
            ret_dict["logprobs"] = log_prob
        
        if compute_entropy:
            entropy = normal.entropy()
            ret_dict["entropy"] = entropy
        
        return ret_dict
    
    def predict_action_batch(
        self,
        env_obs: dict,
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
            
        Returns:
            chunk_actions: numpy array [batch_size, num_action_chunks, action_dim]
            result: dict with prev_logprobs, prev_values, forward_inputs
        """
        obs = self.preprocess_obs(env_obs)
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        if calulate_logprobs:
            actions, log_probs, means = self.get_action(obs)
        else:
            actions = self.get_eval_action(obs)
            log_probs = torch.zeros(obs.shape[0], 1, device=device)
        
        # Reshape to match expected format
        batch_size = obs.shape[0]
        # NOTE: Convert to float32 before numpy conversion (numpy doesn't support bf16)
        actions_float = actions.float() if actions.dtype == torch.bfloat16 else actions
        # chunk_actions = actions_float.reshape(batch_size, self.num_action_chunks, self.action_dim).cpu().numpy()
        chunk_actions = actions_float.reshape(batch_size, 1, self.action_dim).cpu().numpy().repeat(self.num_action_chunks, axis=1)
        
        # Expand log_probs to match action shape
        log_probs_float = log_probs.float() if log_probs.dtype == torch.bfloat16 else log_probs
        # chunk_logprobs = log_probs_float.unsqueeze(-1).expand(-1, self.num_action_chunks, self.action_dim)
        chunk_logprobs = log_probs_float.unsqueeze(-1).expand(-1, 1, self.action_dim).repeat(1, self.num_action_chunks, 1)
        
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": None,  # Residual actor doesn't have value head
            "forward_inputs": {"obs": obs, "action": actions},
        }
        
        return chunk_actions, result

