from typing import Any

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from rlinf.models.embodiment.gr00t.simulation_io import convert_libero_obs_to_gr00t_format


class Gr00tN1d6ForRLActionPrediction(Gr00tPolicy):
    '''
    直接由Isaac-GR00T的Gr00tPolicy类初始化
    - 重写predict_action_batch，作为residual SAC训练流程的接口
    - 本质是wrapper，因此要手动实现部分方法，如cuda, to, eval, parameters，来模拟model的行为
    '''
    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        strict: bool = True,
        num_action_chunks: int = 8,
    ):
        Gr00tPolicy.__init__(
            self,
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        self.num_action_chunks = num_action_chunks

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs=None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        calulate_logprobs=True,
        calulate_values=True,
        return_obs=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Predict action batch from environment observations.
        
        Args:
            env_obs: Environment observation dict with keys:
                - main_images: [B, H, W, C] torch.Tensor
                - wrist_images: [B, H, W, C] torch.Tensor
                - states: [B, 8] torch.Tensor
                - task_descriptions: list[str] of length B
            **kwargs: Additional arguments (ignored for base model)
        
        Returns:
            tuple: (base_chunk_actions, None)
                - base_chunk_actions: [B, num_chunks, action_dim] numpy array
        """
        # Convert env_obs to flat Gr00t format
        flat_obs = convert_libero_obs_to_gr00t_format(env_obs)
        
        # Convert flat format to nested format expected by Gr00tPolicy
        nested_obs = {}
        for modality in ["video", "state", "language"]:
            nested_obs[modality] = {}
            for key in self.modality_configs[modality].modality_keys:
                if modality == "language":
                    # Language is already in correct format from convert_libero_obs_to_gr00t_format
                    # It's a list of strings, need to convert to list[list[str]] (B, 1)
                    lang_key = "annotation.human.action.task_description"
                    if lang_key in flat_obs:
                        nested_obs[modality][key] = [[str(item)] for item in flat_obs[lang_key]]
                else:
                    # Construct flat key (e.g., 'video.image' or 'state.x')
                    flat_key = f"{modality}.{key}"
                    if flat_key in flat_obs:
                        nested_obs[modality][key] = flat_obs[flat_key]
        
        # Get action from Gr00tPolicy
        action_dict, _ = self.get_action(nested_obs)
        
        # Convert action dict to [B, num_chunks, action_dim] numpy array
        # Get action keys in order from modality_config
        # Note: action_dict keys are action keys (e.g., "x", "y"), not "action.x"
        action_keys = self.modality_configs["action"].modality_keys
        action_components = []
        for key in action_keys:
            if key in action_dict:
                # action_dict[key] is [B, T, D] where D is usually 1
                # We need to extract the first num_chunks time steps
                action_component = action_dict[key]  # [B, T, D]
                action_components.append(action_component)
            else:
                raise ValueError(f"Action key '{key}' not found in action_dict")
        
        # Concatenate action components along the last dimension
        # Each component is [B, T, D] where D is usually 1
        # After concatenation: [B, T, action_dim]
        action_array = np.concatenate(action_components, axis=-1)  # [B, T, action_dim]
        
        # Extract first num_chunks time steps
        base_chunk_actions = action_array[:, :self.num_action_chunks, :]  # [B, num_chunks, action_dim]
        
        return base_chunk_actions, None

    #########################################################
    # Overwrite methods to allow access to model attributes
    #########################################################
    def cuda(self):
        self.model.cuda()
        return self
    
    def to(self, device: int | str):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()