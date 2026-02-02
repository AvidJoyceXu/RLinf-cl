import torch
from omegaconf import DictConfig

from gr00t.data.embodiment_tags import EmbodimentTag
from rlinf.models.embodiment.gr00t_n1d6.gr00t_action_model import (
    Gr00tN1d6ForRLActionPrediction,
)


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """
    Get Gr00tN1d6ForRLActionPrediction model for residual policy base model.
    
    Args:
        cfg: Model configuration dict with:
            - model_path: Path to the pretrained model checkpoint
            - embodiment_tag: Embodiment tag (should be "libero_panda")
            - strict: Whether to enforce strict validation (default: True)
        torch_dtype: Torch dtype (ignored, model uses bfloat16 internally)
    
    Returns:
        Gr00tN1d6ForRLActionPrediction: Initialized model instance
    """
    # Get device (default to cuda:0 if available, else cpu)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    # Get embodiment tag
    embodiment_tag_str = cfg.get("embodiment_tag", "libero_panda")
    embodiment_tag = EmbodimentTag(embodiment_tag_str)
    
    # Get model path
    model_path = cfg.get("model_path", "/workspace/hf/libero_spatial_gr00t")
    
    # Get strict flag
    strict = cfg.get("strict", True)
    
    # Create and return model
    model = Gr00tN1d6ForRLActionPrediction(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
        strict=strict,
    )
    
    return model
