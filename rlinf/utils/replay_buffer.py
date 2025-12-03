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


class ReplayBuffer:
    """
    Replay buffer for off-policy RL algorithms (e.g., SAC).
    
    Stores transitions: (obs, next_obs, action, reward, done)
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        device: torch.device,
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            obs_shape: Shape of observations (without batch dimension)
            action_shape: Shape of actions (without batch dimension)
            device: Device to store tensors on
            obs_dtype: Data type for observations
            action_dtype: Data type for actions
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device

        # Pre-allocate buffers
        self.obs = torch.zeros((capacity, *obs_shape), dtype=obs_dtype, device=device)
        self.next_obs = torch.zeros(
            (capacity, *obs_shape), dtype=obs_dtype, device=device
        )
        self.actions = torch.zeros(
            (capacity, *action_shape), dtype=action_dtype, device=device
        )
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        next_obs: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor,
        reward: np.ndarray | torch.Tensor,
        done: np.ndarray | torch.Tensor,
    ):
        """
        Add transition(s) to the replay buffer.

        Args:
            obs: Observation(s). Shape: (batch_size, *obs_shape) or (*obs_shape)
            next_obs: Next observation(s). Shape: (batch_size, *obs_shape) or (*obs_shape)
            action: Action(s). Shape: (batch_size, *action_shape) or (*action_shape)
            reward: Reward(s). Shape: (batch_size,) or scalar
            done: Done flag(s). Shape: (batch_size,) or scalar
        """
        # Convert to tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.from_numpy(next_obs).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward).to(self.device)
        elif isinstance(reward, (int, float)):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        if isinstance(done, np.ndarray):
            done = torch.from_numpy(done).to(self.device)
        elif isinstance(done, bool):
            done = torch.tensor(done, dtype=torch.bool, device=self.device)

        # Handle batch vs single transition
        if obs.ndim == len(self.obs.shape) - 1:
            # Single transition
            batch_size = 1
            obs = obs.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0) if reward.ndim == 0 else reward
            done = done.unsqueeze(0) if done.ndim == 0 else done
        else:
            # Batch of transitions
            batch_size = obs.shape[0]

        # Add to buffer (circular buffer)
        end_ptr = self.ptr + batch_size
        if end_ptr <= self.capacity:
            # Simple case: all fits in one go
            self.obs[self.ptr : end_ptr] = obs
            self.next_obs[self.ptr : end_ptr] = next_obs
            self.actions[self.ptr : end_ptr] = action
            self.rewards[self.ptr : end_ptr] = reward
            self.dones[self.ptr : end_ptr] = done
        else:
            # Wrap around
            first_part = self.capacity - self.ptr
            self.obs[self.ptr :] = obs[:first_part]
            self.next_obs[self.ptr :] = next_obs[:first_part]
            self.actions[self.ptr :] = action[:first_part]
            self.rewards[self.ptr :] = reward[:first_part]
            self.dones[self.ptr :] = done[:first_part]

            second_part = batch_size - first_part
            self.obs[:second_part] = obs[first_part:]
            self.next_obs[:second_part] = next_obs[first_part:]
            self.actions[:second_part] = action[first_part:]
            self.rewards[:second_part] = reward[first_part:]
            self.dones[:second_part] = done[first_part:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing:
                - observations: (batch_size, *obs_shape)
                - next_observations: (batch_size, *obs_shape)
                - actions: (batch_size, *action_shape)
                - rewards: (batch_size,)
                - dones: (batch_size,)
        """
        assert self.size > 0, "Cannot sample from empty replay buffer"
        assert batch_size <= self.size, (
            f"Batch size {batch_size} exceeds buffer size {self.size}"
        )

        # Sample random indices
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "observations": self.obs[indices],
            "next_observations": self.next_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }

    def __len__(self):
        return self.size

