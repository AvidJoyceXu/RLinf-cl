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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import layer_init


class QNetwork(nn.Module):
    """
    Q-network for SAC: Q(s, a) -> scalar value.
    Takes both state and action as input.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        """
        Initialize Q-network.

        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_sizes: Sizes of hidden layers
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build network layers
        layers = []
        input_dim = obs_dim + action_dim

        for hidden_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(input_dim, hidden_size)))
            input_dim = hidden_size

        # Output layer
        layers.append(layer_init(nn.Linear(input_dim, 1), std=1.0))

        self.layers = nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value: Q(s, a).

        Args:
            obs: Observations. Shape: (batch_size, obs_dim) or (batch_size, seq_len, obs_dim)
            action: Actions. Shape: (batch_size, action_dim) or (batch_size, seq_len, action_dim)

        Returns:
            Q-values. Shape: (batch_size,) or (batch_size, seq_len)
        """
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=-1)

        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)

        # Output layer (no activation)
        x = self.layers[-1](x)

        # Squeeze last dimension if needed
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x

