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
from typing import Any, Dict, Optional


class ResidualSACWrapper(nn.Module):
    """
    Wrapper model that combines base model + residual actor for residual SAC.
    
    During rollout: Returns base_action + res_scale * residual_action
    During training: Only residual actor is trained (base model is frozen)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        residual_actor: nn.Module,
        res_scale: float = 1.0,
    ):
        """
        Initialize residual SAC wrapper.
        
        Args:
            base_model: Frozen base model (BC policy)
            residual_actor: Trainable residual actor network
            res_scale: Scaling factor for residual actions
        """
        super().__init__()
        self.base_model = base_model
        self.residual_actor = residual_actor
        self.res_scale = res_scale
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        # Store action dimensions
        self.action_dim = getattr(residual_actor, "action_dim", 7)
        self.num_action_chunks = getattr(residual_actor, "num_action_chunks", 1)
        
        # Dummy config attribute for FSDP compatibility (some FSDP utilities access module.config)
        # Create a simple namespace object
        from types import SimpleNamespace
        self.config = SimpleNamespace(tie_word_embeddings=False)
    
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: Optional[dict] = None,
        calulate_logprobs: bool = True,
        calulate_values: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict actions by combining base + residual.
        
        Args:
            env_obs: Environment observations dict
            calulate_logprobs: Whether to compute log probabilities
            calulate_values: Whether to compute values (from base model if available)
            
        Returns:
            chunk_actions: numpy array [batch_size, num_action_chunks, action_dim]
            result: dict with prev_logprobs, prev_values, forward_inputs
        """
        # Get base actions
        base_actions, base_result = self.base_model.predict_action_batch(
            env_obs=env_obs,
            calulate_logprobs=False,  # Don't need base logprobs for SAC
            calulate_values=calulate_values,
            **kwargs
        )
        
        # Get residual actions
        residual_actions, residual_result = self.residual_actor.predict_action_batch(
            env_obs=env_obs,
            calulate_logprobs=calulate_logprobs,
            base_actions=base_actions,
            calulate_values=False,  # Residual actor doesn't have value head
            **kwargs
        )
        
        # Combine: base + scaled residual
        final_actions = base_actions + self.res_scale * residual_actions
        
        # Prepare result dict
        # Include base_actions in forward_inputs for replay buffer storage
        forward_inputs = residual_result["forward_inputs"].copy()
        
        # Convert base_actions to tensor for storage
        # base_actions is numpy array [batch_size, num_action_chunks, action_dim]
        device = residual_result["forward_inputs"]["obs"].device
        base_actions_tensor = torch.from_numpy(base_actions).to(device)
        residual_actions_tensor = torch.from_numpy(residual_actions).to(device)
        
        # Store base_actions and residual_actions in forward_inputs so they flow through
        # EmbodiedRolloutResult.append_result() which only stores result["forward_inputs"]
        # Shape: [batch_size, num_action_chunks, action_dim]
        forward_inputs["base_actions"] = base_actions_tensor.cpu()
        forward_inputs["residual_actions"] = residual_actions_tensor.cpu()
        
        # Store the observation that was used to query actions
        # This is needed to track (obs, next_obs, base_action, residual_action, base_next_action)
        # The obs here is the one used to query chunk_actions
        if "obs" in forward_inputs:
            # Store the obs that queries actions (for replay buffer)
            forward_inputs["query_obs"] = forward_inputs["obs"].cpu()
        
        result = {
            "prev_logprobs": residual_result["prev_logprobs"],  # Use residual logprobs
            "prev_values": base_result.get("prev_values"),  # Use base values if available
            "forward_inputs": forward_inputs,
        }
        
        return final_actions, result
    
    def forward(
        self,
        obs: torch.Tensor,
        base_actions: torch.Tensor = None,
        obs_use_base_action: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return: action, log_prob
        
        Args:
            obs: Observations, shape [batch_size, obs_dim]
            base_actions: Base actions (optional), shape [batch_size, base_action_dim]
            obs_use_base_action: Whether to concatenate base_actions to obs
        """
        # Prepare observation
        if obs_use_base_action and base_actions is not None:
            obs = torch.cat([obs, base_actions], dim=-1)
        
        action, log_prob, mean_action = self.residual_actor.get_action(obs)
        
        return action, log_prob

        # return self.residual_actor.forward_training(
        #     data=data,
        #     compute_logprobs=compute_logprobs,
        #     compute_entropy=compute_entropy,
        #     compute_values=False,  # Residual actor doesn't have value head
        #     **kwargs
        # )
        
    def load_state_dict(self, state_dict: dict, strict: bool = False):
        """
        Load state dict (only residual actor).
        
        Args:
            state_dict: State dict to load (should only contain residual_actor parameters)
            strict: Whether to strictly enforce matching keys (default False to allow partial state_dicts)
        """
        # Filter state_dict to only include residual_actor parameters
        # Handle both cases: keys with 'residual_actor.' prefix and keys without prefix
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('residual_actor.'):
                # Remove 'residual_actor.' prefix
                filtered_key = key[len('residual_actor.'):]
                filtered_state_dict[filtered_key] = value
            elif not key.startswith('base_model.'):
                # Include keys that don't belong to base_model (should be residual_actor params)
                filtered_state_dict[key] = value
        
        return self.residual_actor.load_state_dict(filtered_state_dict, strict=strict)
    
    def setup_config_and_processor(self, model_config, cfg, input_processor):
        """
        Setup config and processor for the base model.
        This is called by rollout workers to initialize model attributes like max_prompt_length.
        
        Args:
            model_config: Model configuration
            cfg: Full configuration dict
            input_processor: Input processor for the model
        """
        # Forward setup to base_model (which needs attributes like max_prompt_length)
        if hasattr(self.base_model, 'setup_config_and_processor'):
            self.base_model.setup_config_and_processor(model_config, cfg, input_processor)
    
    # Note: We do NOT override state_dict() here because FSDP needs access to all parameters
    # (both base_model and residual_actor) when wrapping. The state_dict will be filtered
    # in get_model_state_dict() or sync_model_to_rollout() to only include residual_actor
    # parameters when syncing to rollout workers.
    #
    # We also do NOT override parameters() or named_parameters() here.
    # This allows FSDP to see and wrap both base_model and residual_actor parameters.
    # The optimizer will automatically exclude base_model parameters because they have
    # requires_grad=False (set in __init__). This enables memory-efficient sharding
    # of the base_model while only training the residual_actor.

