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

"""
Embodiment tags for RLinf GR00T integration.
This module re-exports EmbodimentTag from IsaacGR00T n1d6 and provides
mapping functions for backward compatibility.
"""

# Import from n1d6 implementation
from IsaacGR00T.gr00t_n1d6.data.embodiment_tags import EmbodimentTag

# Mapping from old RLinf tags to n1d6 tags
_RLINF_TO_N1D6_TAG_MAPPING = {
    "libero_franka": "libero_panda",
    "maniskill_widowx": "oxe_widowx",  # Map to closest equivalent
}

# Reverse mapping for reference
_N1D6_TO_RLINF_TAG_MAPPING = {
    "libero_panda": "libero_franka",
    "oxe_widowx": "maniskill_widowx",
}


def map_rlinf_tag_to_n1d6(tag: str) -> str:
    """
    Map RLinf embodiment tag to n1d6 embodiment tag.
    
    Args:
        tag: RLinf embodiment tag string (e.g., "libero_franka")
        
    Returns:
        n1d6 embodiment tag string (e.g., "libero_panda")
    """
    return _RLINF_TO_N1D6_TAG_MAPPING.get(tag, tag)


def get_embodiment_tag(tag: str | EmbodimentTag) -> EmbodimentTag:
    """
    Get EmbodimentTag enum from string or enum, handling RLinf to n1d6 mapping.
    
    Args:
        tag: Embodiment tag as string or EmbodimentTag enum
        
    Returns:
        EmbodimentTag enum instance
    """
    if isinstance(tag, EmbodimentTag):
        return tag
    
    # Map RLinf tags to n1d6 tags
    mapped_tag = map_rlinf_tag_to_n1d6(tag)
    
    # Try to get the enum value
    try:
        return EmbodimentTag(mapped_tag)
    except ValueError:
        # If mapping fails, try the original tag
        return EmbodimentTag(tag)


# Embodiment tag string: to projector index in the Action Expert Module
# This mapping is from the n1d6 processing file
EMBODIMENT_TAG_MAPPING = {
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "new_embodiment": 10,
}
