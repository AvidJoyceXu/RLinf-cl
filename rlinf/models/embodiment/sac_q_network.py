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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-network (critic) for SAC.
    Takes observations and actions as input, outputs Q-values.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        """
        Initialize Q-network.
        
        Args:
            obs_dim: Dimension of observations
            action_dim: Dimension of actions
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        # Input: obs + action
        input_dim = obs_dim + action_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output: single Q-value
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations, shape [batch_size, obs_dim]
            action: Actions, shape [batch_size, action_dim]
            
        Returns:
            Q-values, shape [batch_size, 1]
        """
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=-1)
        q_value = self.network(x)
        return q_value.squeeze(-1)  # [batch_size]


class ResidualQNetwork(nn.Module):
    """
    Q-network for residual SAC that can handle different action input modes:
    - 'res': Only residual actions
    - 'sum': Base + scaled residual (sum)
    - 'concat': Concatenated [residual, base]
    """
    
    def __init__(
        self,
        obs_dim: int,
        residual_action_dim: int,
        base_action_dim: int = 7,
        critic_input: str = "res",
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        """
        Initialize residual Q-network.
        
        Args:
            obs_dim: Dimension of observations
            residual_action_dim: Dimension of residual actions
            base_action_dim: Dimension of base actions
            critic_input: Input mode - 'res', 'sum', or 'concat'
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.residual_action_dim = residual_action_dim
        self.base_action_dim = base_action_dim
        self.critic_input = critic_input
        
        # Determine action dimension based on input mode
        if critic_input == "res":
            action_dim = residual_action_dim
        elif critic_input == "sum":
            action_dim = base_action_dim  # Sum has same dim as base
        elif critic_input == "concat":
            action_dim = residual_action_dim + base_action_dim
        else:
            raise ValueError(f"Unknown critic_input mode: {critic_input}")
        
        # Build network layers
        layers = []
        input_dim = obs_dim + action_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        obs: torch.Tensor,
        residual_action: torch.Tensor,
        base_action: Optional[torch.Tensor] = None,
        res_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations, shape [batch_size, obs_dim]
            residual_action: Residual actions, shape [batch_size, residual_action_dim]
            base_action: Base actions, shape [batch_size, base_action_dim] (required for 'sum' and 'concat')
            res_scale: Scaling factor for residual actions (used for 'sum' mode)
            
        Returns:
            Q-values, shape [batch_size]
        """
        if self.critic_input == "res":
            action = residual_action
        elif self.critic_input == "sum":
            if base_action is None:
                raise ValueError("base_action required for 'sum' mode")
            action = base_action + res_scale * residual_action
        elif self.critic_input == "concat":
            if base_action is None:
                raise ValueError("base_action required for 'concat' mode")
            action = torch.cat([residual_action, base_action], dim=-1)
        else:
            raise ValueError(f"Unknown critic_input mode: {self.critic_input}")
        
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=-1)
        q_value = self.network(x)
        return q_value.squeeze(-1)  # [batch_size]

