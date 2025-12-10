import numpy as np
import torch
from typing import Optional, Tuple


class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL algorithms like SAC.
    
    Stores transitions: (obs, next_obs, action, reward, done)
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        device: torch.device,
        n_envs: int = 1,
        handle_timeout_termination: bool = True,
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            obs_shape: Shape of observations (without batch dimension)
            action_shape: Shape of actions (without batch dimension)
            device: Device to store tensors on
            n_envs: Number of parallel environments
            handle_timeout_termination: Whether to handle timeout terminations specially
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.handle_timeout_termination = handle_timeout_termination
        
        # Initialize buffers
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Optional[dict] = None,
    ):
        """
        Add transitions to the replay buffer.
        
        Args:
            obs: Observations, shape [n_envs, *obs_shape]
            next_obs: Next observations, shape [n_envs, *obs_shape]
            action: Actions, shape [n_envs, *action_shape]
            reward: Rewards, shape [n_envs]
            done: Done flags, shape [n_envs]
            infos: Optional info dict
        """
        # Handle batched inputs
        n_transitions = obs.shape[0] if len(obs.shape) > len(self.obs_shape) else 1
        
        if n_transitions == 1 and len(obs.shape) == len(self.obs_shape):
            # Single transition
            obs = obs[np.newaxis, ...]
            next_obs = next_obs[np.newaxis, ...]
            action = action[np.newaxis, ...] if len(action.shape) == len(self.action_shape) else action
            reward = np.array([reward]) if np.isscalar(reward) else reward
            done = np.array([done]) if np.isscalar(done) else done
        
        for i in range(n_transitions):
            self.observations[self.ptr] = obs[i]
            self.next_observations[self.ptr] = next_obs[i]
            self.actions[self.ptr] = action[i] if len(action.shape) > len(self.action_shape) else action
            self.rewards[self.ptr] = reward[i] if len(reward.shape) > 0 else reward
            self.dones[self.ptr] = done[i] if len(done.shape) > 0 else done
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (observations, next_observations, actions, rewards, dones) as torch tensors
        """
        assert self.size > 0, "Cannot sample from empty replay buffer"
        
        idx = np.random.randint(0, self.size, size=batch_size)
        
        obs = torch.FloatTensor(self.observations[idx]).to(self.device)
        next_obs = torch.FloatTensor(self.next_observations[idx]).to(self.device)
        actions = torch.FloatTensor(self.actions[idx]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idx]).to(self.device)
        dones = torch.BoolTensor(self.dones[idx]).to(self.device)

        # Ensure batch dimension is preserved (important when batch_size=1)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if next_obs.ndim == 1:
            next_obs = next_obs.unsqueeze(0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if rewards.ndim == 0:
            rewards = rewards.unsqueeze(0)
        if dones.ndim == 0:
            dones = dones.unsqueeze(0)
        
        return obs, next_obs, actions, rewards, dones
    
    def __len__(self) -> int:
        """Return current size of the replay buffer."""
        return self.size

