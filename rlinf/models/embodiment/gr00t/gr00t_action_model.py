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

import json
import random
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from IsaacGR00T.gr00t_n1d6.data.embodiment_tags import EmbodimentTag
from rlinf.models.embodiment.gr00t.schema import DatasetMetadata, ModalityConfig
from rlinf.models.embodiment.gr00t.transform_wrapper import ComposedModalityTransform
# from IsaacGR00T.gr00t_n1d6.model.action_head.flow_matching_action_head import (
#     FlowmatchingActionHead,
#     FlowmatchingActionHeadConfig,
# )
# Import from local Isaac-GR00T implementation
from IsaacGR00T.gr00t_n1d6.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6ActionHead
from IsaacGR00T.gr00t_n1d6.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from torch.distributions import Normal
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


class Gr00tN1d6ActionHeadForRLActionPrediction(Gr00tN1d6ActionHead):
    """Wrapper for Gr00tN1d6ActionHead with RL-specific functionality."""
    
    def __init__(
        self,
        config: Gr00tN1d6Config,
        rl_head_config: dict[str, Any],
        output_action_chunks: int,
        valid_action_dim: int,
    ):
        super().__init__(config)
        self.action_chunk = output_action_chunks
        self.rl_config = rl_head_config
        self.padding_value = rl_head_config.padding_value
        self.valid_action_dim = valid_action_dim
        # Add global_step for noise annealing (initialized to 0, should be updated during training)
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

        if self.rl_config.use_vlm_value:
            proj_width = 2048
        else:
            proj_width = 3584

        if self.rl_config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                out_dim=self.config.max_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    def get_logprob_norm(self, sample, mu, sigma):
        if self.rl_config.safe_get_logprob:
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        Pay attention: The time notation of gr00t is different from openpi.
        In gr00t, the time is from 0 to 1, while in openpi, the time is from 1 to 0.
        """
        # expand the shape
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.rl_config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.rl_config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.rl_config.noise_level).to(device)

        # velocity prediction
        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        # Note: Gr00tN1d6 does not use future_tokens, so we directly concatenate state and action features
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Run model forward - need to handle AlternateVLDiT vs DiT
        if self.config.use_alternate_vl_dit:
            # Need to get image_mask and backbone_attention_mask from somewhere
            # For now, create dummy masks if not available
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
        model_output = model_output[:, -self.action_horizon :]

        # ode/sde sampling
        v_t = self.action_decoder(model_output, embodiment_id)

        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        # Emphasize: In Gr00t, x0: noise, x1: data.
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = (
                t_input + delta
            )  # notice the plus here, it's different from openpi.
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.rl_config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.rl_config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = (torch.ones_like(t_input) - (t_input + delta)) * cos_term
                x1_weight = t_input + delta
                x_t_std = (1 - (t_input + delta)) * sin_term
            elif self.rl_config.noise_method == "reinflow":
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = self.reinflow_explore_noise_net(model_output)
            else:
                raise ValueError(f"Invalid noise method: {self.rl_config.noise_method}")
        # In eval, this equals to x_t_mean = x_t + v*dt(dt>0).
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        # determine the denoise step for the logprob calculation
        if mode == "train":
            if self.rl_config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.rl_config.noise_method == "flow_sde":
                    if self.rl_config.ignore_last:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
                elif self.rl_config.noise_method == "flow_cps":
                    # the last denoising step of the flow-cps is deterministic
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
                elif self.rl_config.noise_method == "reinflow":
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        # Run denoising steps.
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="train",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )
            else:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="eval",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.valid_action_dim
        ]
        if compute_values:
            values = self.get_value(vl_embs, state_features)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(
            data={"action_pred": x_0}
        ), {  # this is for gr00t validity check
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]

        chains_log_probs = []

        if self.rl_config.joint_logprob:
            num_steps = self.config.num_steps if hasattr(self.config, 'num_steps') else 1
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )  # (B, 1)
        return chains_log_probs, chains_values

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def get_value(self, vl_embs, state_features):
        # TODO: add value vlm mode param
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.value_vlm_mode == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :]
        vl_embs_value = vl_embs_value.mean(dim=1, keepdim=False)
        # vl_embs_value = vl_embs_value.to(dtype=torch.float32)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.use_vlm_value:
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
        values_vlm = self.value_head(value_embs)[:, 0]
        return values_vlm


# class FlowMatchingActionHeadForRLActionPrediction(FlowmatchingActionHead):
#     def __init__(
#         self,
#         config: FlowmatchingActionHeadConfig,
#         rl_head_config: dict[str, Any],
#         output_action_chunks: int,
#         valid_action_dim: int,
#     ):
#         super().__init__(config)
#         self.action_chunk = output_action_chunks
#         self.rl_config = rl_head_config
#         self.padding_value = rl_head_config.padding_value
#         self.valid_action_dim = valid_action_dim
#         # Add global_step for noise annealing if not already present
#         if not hasattr(self, 'global_step'):
#             self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

#         if self.rl_config.use_vlm_value:
#             proj_width = 2048
#         else:
#             proj_width = 3584

#         if self.rl_config.add_value_head:
#             self.value_head = ValueHead(
#                 input_dim=proj_width,
#                 hidden_sizes=(1024, 512, 256),
#                 output_dim=1,
#                 activation="relu",
#                 bias_last=True,
#             )

#         if self.rl_config.noise_method == "reinflow":
#             self.reinflow_explore_noise_net = ExploreNoiseNet(
#                 in_dim=self.hidden_size,
#                 out_dim=self.config.action_dim,
#                 hidden_dims=[128, 64],
#                 activation_type="tanh",
#                 noise_logvar_range=[0.08, 0.16],
#                 noise_scheduler_type="learn",
#             )

#     def get_logprob_norm(self, sample, mu, sigma):
#         if self.rl_config.safe_get_logprob:
#             dist = Normal(loc=mu, scale=sigma)
#             return dist.log_prob(sample)
#         else:
#             # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
#             mask = sigma == 0
#             sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
#             constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
#                 2 * torch.pi * torch.ones_like(sample)
#             )
#             exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
#             log_prob = constant_term + exponent_term
#             log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
#             return log_prob

#     def sample_mean_var_val(
#         self,
#         vl_embs: torch.Tensor,
#         denoise_steps: int,
#         x_t: torch.Tensor,
#         embodiment_id: int,
#         state_features: torch.Tensor,
#         idx: Optional[int | torch.Tensor],
#         mode: Literal["train", "eval"] = "train",
#         compute_values=False,
#     ):
#         """
#         Sample the mean, variance and value of the action at a given timestep.
#         Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
#         Pay attention: The time notation of gr00t is different from openpi.
#         In gr00t, the time is from 0 to 1, while in openpi, the time is from 1 to 0.
#         """
#         # expand the shape
#         bsize = vl_embs.shape[0]
#         device = vl_embs.device
#         if isinstance(idx, int):
#             idx = torch.tensor(idx).expand(bsize)
#         # build parameters
#         if self.rl_config.noise_anneal:
#             # noise annealing
#             noise_start, noise_end, anneal_steps = self.rl_config.noise_params
#             noise_level = (
#                 noise_start
#                 + (noise_end - noise_start)
#                 * min(self.global_step, anneal_steps)
#                 / anneal_steps
#             )
#             noise_level = torch.tensor(noise_level).to(device)
#         else:
#             # fixed noise level
#             noise_level = torch.tensor(self.rl_config.noise_level).to(device)

#         # velocity prediction
#         t_cont = idx / float(denoise_steps)
#         timesteps_tensor = (
#             (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
#         )
#         action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)
#         # Maybe add position embedding.
#         if self.config.add_pos_embed:
#             pos_ids = torch.arange(
#                 action_features.shape[1], dtype=torch.long, device=device
#             )
#             pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
#             action_features = action_features + pos_embs

#         # Join vision, language, state and action embedding along sequence dimension.
#         future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
#             vl_embs.shape[0], -1, -1
#         )
#         sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

#         model_output = self.model(
#             hidden_states=sa_embs,
#             encoder_hidden_states=vl_embs,
#             timestep=timesteps_tensor,
#         )
#         model_output = model_output[:, -self.action_horizon :]

#         # ode/sde sampling
#         v_t = self.action_decoder(model_output, embodiment_id)

#         timesteps = torch.linspace(
#             0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
#         )
#         t_input = timesteps[idx]
#         delta = timesteps[idx + 1] - timesteps[idx]
#         delta = delta[:, None, None].expand_as(x_t)
#         t_input = t_input[:, None, None].expand_as(x_t)
#         # Emphasize: In Gr00t, x0: noise, x1: data.
#         x0_pred = x_t - v_t * t_input
#         x1_pred = x_t + v_t * (1 - t_input)

#         if mode == "eval":
#             x0_weight = 1 - (t_input + delta)
#             x1_weight = (
#                 t_input + delta
#             )  # notice the plus here, it's different from openpi.
#             x_t_std = torch.zeros_like(t_input)
#         elif mode == "train":
#             if self.rl_config.noise_method == "flow_sde":
#                 sigmas = (
#                     noise_level
#                     * torch.sqrt(
#                         (1 - timesteps)
#                         / torch.where(timesteps == 0, timesteps[1], timesteps)
#                     )[:-1]
#                 )
#                 sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
#                 x0_weight = (
#                     torch.ones_like(t_input)
#                     - (t_input + delta)
#                     - sigma_i**2 * delta / (2 * (1 - t_input))
#                 )
#                 x1_weight = t_input + delta
#                 x_t_std = torch.sqrt(delta) * sigma_i
#             elif self.rl_config.noise_method == "flow_cps":
#                 pi = torch.pi
#                 cos_term = torch.cos(pi * noise_level / 2).to(device)
#                 sin_term = torch.sin(pi * noise_level / 2).to(device)
#                 x0_weight = (torch.ones_like(t_input) - (t_input + delta)) * cos_term
#                 x1_weight = t_input + delta
#                 x_t_std = (1 - (t_input + delta)) * sin_term
#             elif self.rl_config.noise_method == "reinflow":
#                 x0_weight = 1 - (t_input + delta)
#                 x1_weight = t_input + delta
#                 x_t_std = self.reinflow_explore_noise_net(model_output)
#             else:
#                 raise ValueError(f"Invalid noise method: {self.rl_config.noise_method}")
#         # In eval, this equals to x_t_mean = x_t + v*dt(dt>0).
#         x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
#         return x_t_mean, x_t_std

#     def get_rl_action(
#         self,
#         backbone_output: BatchFeature,
#         action_input: BatchFeature,
#         mode: Literal["train", "eval"] = "train",
#         compute_values=True,
#     ) -> BatchFeature:
#         backbone_output = self.process_backbone_output(backbone_output)
#         # Get vision and language embeddings.
#         vl_embs = backbone_output.backbone_features
#         embodiment_id = action_input.embodiment_id
#         # Embed state.
#         state_features = self.state_encoder(action_input.state, embodiment_id)
#         # Set initial actions as the sampled noise.
#         batch_size = vl_embs.shape[0]
#         device = vl_embs.device
#         x_t = torch.randn(
#             size=(batch_size, self.config.action_horizon, self.config.action_dim),
#             dtype=vl_embs.dtype,
#             device=device,
#         )

#         chains = [x_t]
#         log_probs = []

#         if self.rl_config.joint_logprob:
#             initial_log_prob = self.get_logprob_norm(
#                 x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
#             )
#             log_probs.append(initial_log_prob)

#         num_steps = self.num_inference_timesteps
#         # determine the denoise step for the logprob calculation
#         if mode == "train":
#             if self.rl_config.joint_logprob:
#                 denoise_inds = torch.arange(num_steps)
#             else:
#                 if self.rl_config.noise_method == "flow_sde":
#                     if self.rl_config.ignore_last:
#                         denoise_inds = torch.tensor(
#                             [random.randint(0, num_steps - 2)] * num_steps
#                         )
#                     else:
#                         denoise_inds = torch.tensor(
#                             [random.randint(0, num_steps - 1)] * num_steps
#                         )
#                 elif self.rl_config.noise_method == "flow_cps":
#                     # the last denoising step of the flow-cps is deterministic
#                     denoise_inds = torch.tensor(
#                         [random.randint(0, num_steps - 1)] * num_steps
#                     )
#                 elif self.rl_config.noise_method == "reinflow":
#                     denoise_inds = torch.tensor(
#                         [random.randint(0, num_steps - 1)] * num_steps
#                     )
#         else:
#             denoise_inds = torch.tensor([-1] * num_steps)
#         denoise_inds = denoise_inds[None].repeat(batch_size, 1)

#         # Run denoising steps.
#         for idx in range(num_steps):
#             if idx == denoise_inds[0][idx]:
#                 x_t_mean, x_t_std = self.sample_mean_var_val(
#                     vl_embs=vl_embs,
#                     idx=idx,
#                     x_t=x_t,
#                     embodiment_id=embodiment_id,
#                     state_features=state_features,
#                     mode="train",
#                     denoise_steps=num_steps,
#                     compute_values=compute_values,
#                 )
#             else:
#                 x_t_mean, x_t_std = self.sample_mean_var_val(
#                     vl_embs=vl_embs,
#                     idx=idx,
#                     x_t=x_t,
#                     embodiment_id=embodiment_id,
#                     state_features=state_features,
#                     mode="eval",
#                     denoise_steps=num_steps,
#                     compute_values=compute_values,
#                 )

#             x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
#             log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

#             chains.append(x_t)
#             log_probs.append(log_prob)

#         x_0 = x_t
#         chains = torch.stack(chains, dim=1)
#         log_probs = torch.stack(log_probs, dim=1)[
#             :, :, : self.action_chunk, : self.valid_action_dim
#         ]
#         if compute_values:
#             values = self.get_value(vl_embs, state_features)
#             values = values[:, None]
#         else:
#             values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

#         return BatchFeature(
#             data={"action_pred": x_0}
#         ), {  # this is for gr00t validity check
#             "actions": x_0,
#             "action_pred": x_0,
#             "chains": chains,
#             "prev_logprobs": log_probs,
#             "prev_values": values,
#             "denoise_inds": denoise_inds,
#         }

#     def forward(
#         self,
#         backbone_output: BatchFeature,
#         action_input: BatchFeature,
#         chains,
#         denoise_inds,
#         compute_values=True,
#     ):
#         backbone_output = self.process_backbone_output(backbone_output)
#         # Get vision and language embeddings.
#         vl_embs = backbone_output.backbone_features
#         embodiment_id = action_input.embodiment_id
#         # Embed state.
#         state_features = self.state_encoder(action_input.state, embodiment_id)
#         # Set initial actions as the sampled noise.
#         batch_size = vl_embs.shape[0]

#         chains_log_probs = []

#         if self.rl_config.joint_logprob:
#             num_steps = self.config.num_steps if hasattr(self.config, 'num_steps') else 1
#             initial_log_prob = self.get_logprob_norm(
#                 chains[:, 0],
#                 torch.zeros_like(chains[:, 0]),
#                 torch.ones_like(chains[:, 0]),
#             )
#             chains_log_probs.append(initial_log_prob)
#         else:
#             num_steps = 1
#         for idx in range(num_steps):
#             denoise_ind = denoise_inds[:, idx]
#             chains_pre = chains[torch.arange(batch_size), denoise_ind]
#             chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
#             x_t_mean, x_t_std = self.sample_mean_var_val(
#                 vl_embs=vl_embs,
#                 idx=denoise_ind,
#                 x_t=chains_pre,
#                 embodiment_id=embodiment_id,
#                 state_features=state_features,
#                 mode="train",
#                 denoise_steps=self.num_inference_timesteps,
#                 compute_values=compute_values,
#             )
#             log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
#             chains_log_probs.append(log_probs)

#         chains_log_probs = torch.stack(chains_log_probs, dim=1)
#         if compute_values:
#             chains_values = self.get_value(vl_embs, state_features)
#             chains_values = chains_values[:, None]
#         else:
#             chains_values = torch.zeros(
#                 (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
#             )  # (B, 1)
#         return chains_log_probs, chains_values

#     def sample_noise(self, shape, device):
#         return torch.normal(
#             mean=0.0,
#             std=1.0,
#             size=shape,
#             dtype=torch.bfloat16,
#             device=device,
#         )

#     def get_value(self, vl_embs, state_features):
#         # TODO: add value vlm mode param
#         bsize = vl_embs.shape[0]
#         mask_length = vl_embs.shape[1]
#         if self.rl_config.value_vlm_mode == "mean_token":
#             prefix_mask = [True] * mask_length
#         elif self.rl_config.value_vlm_mode == "last_token":
#             prefix_mask = [False] * (mask_length - 1) + [True] * 1
#         elif self.rl_config.value_vlm_mode == "first_token":
#             prefix_mask = [True] * 1 + [False] * (mask_length - 1)
#         vl_embs_value = vl_embs[:, prefix_mask, :]
#         vl_embs_value = vl_embs_value.mean(dim=1, keepdim=False)
#         # vl_embs_value = vl_embs_value.to(dtype=torch.float32)
#         state_features_value = state_features.reshape(bsize, -1)
#         if self.rl_config.use_vlm_value:
#             value_embs = vl_embs_value
#         else:
#             value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
#         values_vlm = self.value_head(value_embs)[:, 0]
#         return values_vlm


class GR00T_N1_5_ForRLActionPrediction(BasePolicy, Gr00tN1d6):
    """
    GR00T_N1_5 model for reinforcement learning action prediction.
    It's a combination of the Gr00tPolicy and Gr00tN1d6 model.
    
    Note: This class name is kept for backward compatibility, but it now uses Gr00tN1d6.

    Notes:
        - Device is handled by huggingface worker.
        - EmbodimentTag determines the state encoder and action head to use.
          we use "new_embodiment" reserved by gr00t.

    """

    _no_split_modules = [
        "Eagle2_5_VLForConditionalGeneration",
        "Gr00tN1d6ActionHeadForRLActionPrediction",
        "TimestepEncoder",
        "TimestepEmbedding",
        "ValueHead",
    ]

    def __init__(
        self,
        config: Gr00tN1d6Config,
        rl_head_config: dict[str, Any],
        local_model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        # Initialize Gr00tN1d6 with transformers_loading_kwargs instead of local_model_path
        Gr00tN1d6.__init__(self, config, transformers_loading_kwargs=transformers_loading_kwargs)

        self.padding_value = rl_head_config.padding_value
        self._modality_config = modality_config  # ModalityConfig(delta_indices=[0], modality_keys=['video.ego_view'])
        self._modality_transform = modality_transform
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks
        self.model_path = Path(local_model_path)

        # Convert string embodiment tag to EmbodimentTag enum if needed
        # Handle mapping from RLinf tags to n1d6 tags
        from rlinf.models.embodiment.gr00t.embodiment_tags import get_embodiment_tag
        self.embodiment_tag = get_embodiment_tag(embodiment_tag)

        if denoising_steps is not None:
            if hasattr(self, "action_head") and hasattr(
                self.action_head, "num_inference_timesteps"
            ):
                self.action_head.num_inference_timesteps = denoising_steps

        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION[obs_converter_type]
        self._load_metadata(self.model_path / "experiment_cfg")

        # Replace action head with RL-specific wrapper
        # Note: The base action_head is already created by Gr00tN1d6.__init__
        # We need to replace it with our RL wrapper
        base_action_head = self.action_head
        self.action_head = Gr00tN1d6ActionHeadForRLActionPrediction(
            config, rl_head_config, output_action_chunks, self.valid_action_dim
        )
        # Copy weights from base action head if it exists
        # This allows loading pretrained weights
        if hasattr(base_action_head, 'state_dict'):
            try:
                self.action_head.load_state_dict(base_action_head.state_dict(), strict=False)
            except Exception as e:
                print(f"Warning: Could not load action head weights: {e}")

    def eval(self):
        self._modality_transform.eval()
        Gr00tN1d6.eval(self)

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        normalized_input = {
            "state": data["state"],
            "state_mask": data["state_mask"],
            "eagle_input_ids": data["eagle_input_ids"],
            "eagle_attention_mask": data["eagle_attention_mask"],
            "eagle_pixel_values": data["eagle_pixel_values"].reshape(
                -1, *data["eagle_pixel_values"].shape[2:]
            ),
            "eagle_image_sizes": data["eagle_image_sizes"].reshape(
                -1, *data["eagle_image_sizes"].shape[2:]
            ),
            "embodiment_id": data["embodiment_id"],
        }
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        # post process
        if self.action_head.rl_config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = data["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = data["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": None,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ):
        # Here we have a source causing tiny inference-training inconsistency,
        # force convert the state to bf16 then back to float32 to reproduce the info loss in training.
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()

        observations = self.obs_convert_fn(env_obs)
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)

        for key in normalized_input:
            if normalized_input[key].dtype == torch.float32:
                normalized_input[key] = normalized_input[key].to(torch.bfloat16)

        normalized_input["eagle_input_ids"] = torch.nn.functional.pad(
            normalized_input["eagle_input_ids"],
            pad=(0, self.padding_value - normalized_input["eagle_input_ids"].shape[-1]),
            mode="constant",
            value=0,
        )
        normalized_input["eagle_attention_mask"] = torch.nn.functional.pad(
            normalized_input["eagle_attention_mask"],
            pad=(
                0,
                self.padding_value - normalized_input["eagle_attention_mask"].shape[-1],
            ),
            mode="constant",
            value=0,
        )

        normalized_action, result = self._get_rl_action(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )

        return raw_action, result

    def apply_transforms(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        obs = self._modality_transform(obs)
        if obs["eagle_pixel_values"].ndim == 4:
            obs["eagle_pixel_values"] = obs["eagle_pixel_values"].unsqueeze(1)
        return obs

    def unapply_transforms(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def _get_rl_action(self, normalized_input: dict[str, Any]) -> torch.Tensor:
        # We expand get_action() and replace action head inference with RL inference.
        # Convert eagle_* keys to standard keys expected by backbone
        backbone_normalized_input = normalized_input.copy()
        if "eagle_input_ids" in backbone_normalized_input:
            backbone_normalized_input["input_ids"] = backbone_normalized_input.pop("eagle_input_ids")
        if "eagle_attention_mask" in backbone_normalized_input:
            backbone_normalized_input["attention_mask"] = backbone_normalized_input.pop("eagle_attention_mask")
        if "eagle_pixel_values" in backbone_normalized_input:
            backbone_normalized_input["pixel_values"] = backbone_normalized_input.pop("eagle_pixel_values")
        
        backbone_inputs, action_inputs = self.prepare_input(backbone_normalized_input)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
            backbone_outputs, action_inputs
        )
        actions = rlinf_outputs["actions"]
        # Note: validate_data may not exist in Gr00tN1d6, so we check first
        if hasattr(self, 'validate_data'):
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        actions = actions.float()

        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        bsize = normalized_input["state"].shape[0]
        forward_inputs["eagle_pixel_values"] = normalized_input[
            "eagle_pixel_values"
        ].reshape(
            bsize, self.image_nums, *normalized_input["eagle_pixel_values"].shape[1:]
        )
        forward_inputs["eagle_image_sizes"] = normalized_input[
            "eagle_image_sizes"
        ].reshape(
            bsize, self.image_nums, *normalized_input["eagle_image_sizes"].shape[1:]
        )

        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        return actions, result

    def _get_action_from_normalized_input(
        self, normalized_input: dict[str, Any]
    ) -> torch.Tensor:
        # Set up autocast context if needed
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=self.compute_dtype),
        ):
            model_pred = self.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(
        self, normalized_action: torch.Tensor
    ) -> dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        # metadata_path = exp_cfg_dir / "metadata.json"
        metadata_path = exp_cfg_dir / "dataset_statistics.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        # n1d6 format: {embodiment_tag: {modality: {key: {stat_type: values}}}}
        n1d6_stats = metadatas.get(self.embodiment_tag.value)
        if n1d6_stats is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the dataset_statistics.json file is present at {metadata_path}",
            )

        # Create DatasetMetadata from n1d6 format
        rlinf_metadata = DatasetMetadata.from_n1d6_format(
            n1d6_stats=n1d6_stats,
            embodiment_tag=self.embodiment_tag,
            modality_config=self._modality_config,
        )

        # Convert to gr00t format for transform compatibility
        # gr00t's transform system uses isinstance checks that require gr00t classes
        from rlinf.models.embodiment.gr00t.metadata_adapter import convert_to_gr00t_metadata
        
        gr00t_metadata = convert_to_gr00t_metadata(rlinf_metadata)
        self._modality_transform.set_metadata(gr00t_metadata)
        
        # Store both versions - rlinf version for our use, gr00t version for transforms
        self.metadata = rlinf_metadata
        self._gr00t_metadata = gr00t_metadata

        # calculate real intput action dim for rl learning.
        valid_action_dim = 0
        for v in rlinf_metadata.modalities.action.values():
            valid_action_dim += v.shape[0]
        self.valid_action_dim = valid_action_dim

        self.image_nums = len(rlinf_metadata.modalities.video.keys())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        torch_dtype: torch.dtype = torch.bfloat16,
        embodiment_tag: Union[str, EmbodimentTag] = None,
        modality_config: dict[str, ModalityConfig] = None,
        modality_transform: ComposedModalityTransform = None,
        denoising_steps: Optional[int] = None,
        output_action_chunks: int = 1,
        obs_converter_type: str = "libero",
        tune_visual: bool = False,
        tune_llm: bool = False,
        rl_head_config: dict[str, Any] = None,
        **kwargs,
    ):
        """
        Load a pretrained model with custom RL-specific arguments.
        
        This method extends the standard from_pretrained to handle RL-specific
        configuration like embodiment_tag, modality_config, etc.
        """
        # Import Gr00tN1d6 and Gr00tN1d6Config to ensure they are registered with transformers
        # This must be done before AutoConfig.from_pretrained to ensure the registration code runs
        # The import at the top of the file may not have executed the registration code yet
        try:
            # Try importing from the already imported module (this will execute registration)
            from IsaacGR00T.gr00t_n1d6.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
            from IsaacGR00T.gr00t_n1d6.configs.model.gr00t_n1d6 import Gr00tN1d6Config
        except (ImportError, AttributeError):
            # If that fails, try alternative import paths
            try:
                import sys
                isaac_path = Path(__file__).parent.parent.parent.parent / "IsaacGR00T"
                if str(isaac_path) not in sys.path:
                    sys.path.insert(0, str(isaac_path))
                from gr00t_n1d6.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
                from gr00t_n1d6.configs.model.gr00t_n1d6 import Gr00tN1d6Config
            except ImportError:
                # Last resort: use the already imported classes
                pass  # Gr00tN1d6 and Gr00tN1d6Config are already imported at module level
        
        from transformers import AutoConfig
        
        # Load config from pretrained model
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **{k: v for k, v in kwargs.items() if k not in ['trust_remote_code']}
        )
        
        # Convert config to Gr00tN1d6Config if needed
        if not isinstance(config, Gr00tN1d6Config):
            # Convert PretrainedConfig object to dict
            if isinstance(config, dict):
                config_dict = config
            else:
                # PretrainedConfig objects have to_dict() method
                config_dict = config.to_dict()
            
            # Filter out fields that are not in Gr00tN1d6Config class definition
            # This avoids TypeError when passing unknown fields to dataclass __init__
            valid_fields = set(Gr00tN1d6Config.__dataclass_fields__.keys())
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
            
            # Create Gr00tN1d6Config instance with filtered dictionary
            config = Gr00tN1d6Config(**filtered_dict)
        
        # Update config with tuning parameters
        config.tune_visual = tune_visual
        config.tune_llm = tune_llm
        
        # Prepare transformers loading kwargs
        transformers_loading_kwargs = {
            "trust_remote_code": True,
            **{k: v for k, v in kwargs.items() if k.startswith('transformers_') or k in ['trust_remote_code']}
        }
        
        # Load the base model using Gr00tN1d6.from_pretrained directly
        # This ensures the model is loaded correctly even if AutoModel registration hasn't completed
        model = Gr00tN1d6.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            transformers_loading_kwargs=transformers_loading_kwargs,
            **{k: v for k, v in kwargs.items() if k not in ['config', 'torch_dtype', 'trust_remote_code', 'transformers_loading_kwargs'] and not k.startswith('transformers_')}
        )
        
        # Convert to our custom class
        model.__class__ = cls
        
        # Set custom attributes
        # Handle mapping from RLinf tags to n1d6 tags
        from rlinf.models.embodiment.gr00t.embodiment_tags import get_embodiment_tag
        model.embodiment_tag = get_embodiment_tag(embodiment_tag)
        model._modality_config = modality_config
        model._modality_transform = modality_transform
        model.compute_dtype = torch_dtype
        model.output_action_chunks = output_action_chunks
        model.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        model.action_convert_fn = ACTION_CONVERSION[obs_converter_type]
        
        # Load metadata
        model_path = Path(pretrained_model_name_or_path)
        model._load_metadata(model_path / "experiment_cfg")
        
        # Replace action head with RL-specific wrapper
        if rl_head_config is not None:
            base_action_head = model.action_head
            model.action_head = Gr00tN1d6ActionHeadForRLActionPrediction(
                config, rl_head_config, output_action_chunks, model.valid_action_dim
            )
            # Set padding_value from rl_head_config
            model.padding_value = rl_head_config.padding_value
            # Try to load weights from base action head
            if hasattr(base_action_head, 'state_dict'):
                try:
                    model.action_head.load_state_dict(base_action_head.state_dict(), strict=False)
                except Exception as e:
                    print(f"Warning: Could not load action head weights: {e}")
        
        # Set denoising steps if provided
        if denoising_steps is not None:
            if hasattr(model, "action_head") and hasattr(
                model.action_head, "num_inference_timesteps"
            ):
                model.action_head.num_inference_timesteps = denoising_steps
        
        return model
