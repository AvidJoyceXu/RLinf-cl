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
Adapter to convert RLinf DatasetMetadata to gr00t DatasetMetadata format.
This is needed because gr00t's transform system uses isinstance checks
that require the exact gr00t classes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlinf.models.embodiment.gr00t.schema import DatasetMetadata as RLinfDatasetMetadata

# Import gr00t classes for conversion
try:
    from gr00t.data.schema import (
        DatasetMetadata as Gr00tDatasetMetadata,
        DatasetModalities as Gr00tDatasetModalities,
        DatasetStatistics as Gr00tDatasetStatistics,
        DatasetStatisticalValues as Gr00tDatasetStatisticalValues,
        StateActionMetadata as Gr00tStateActionMetadata,
        VideoMetadata as Gr00tVideoMetadata,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag as Gr00tEmbodimentTag
except ImportError:
    # Fallback if gr00t is not available
    Gr00tDatasetMetadata = None
    Gr00tDatasetModalities = None
    Gr00tDatasetStatistics = None
    Gr00tDatasetStatisticalValues = None
    Gr00tStateActionMetadata = None
    Gr00tVideoMetadata = None
    Gr00tEmbodimentTag = None


def convert_to_gr00t_metadata(
    rlinf_metadata: "RLinfDatasetMetadata",
) -> "Gr00tDatasetMetadata":
    """
    Convert RLinf DatasetMetadata to gr00t DatasetMetadata format.
    
    This function creates gr00t-compatible metadata objects that will pass
    isinstance checks in gr00t's transform system.
    
    Args:
        rlinf_metadata: RLinf DatasetMetadata instance
        
    Returns:
        gr00t DatasetMetadata instance compatible with gr00t transforms
    """
    if Gr00tDatasetMetadata is None:
        raise ImportError(
            "gr00t package is required for metadata conversion. "
            "Please ensure gr00t is installed in your environment."
        )
    
    # Convert statistical values
    gr00t_stat_values_state = {}
    for key, stat_val in rlinf_metadata.statistics.state.items():
        gr00t_stat_values_state[key] = Gr00tDatasetStatisticalValues(
            max=stat_val.max,
            min=stat_val.min,
            mean=stat_val.mean,
            std=stat_val.std,
            q01=stat_val.q01,
            q99=stat_val.q99,
        )
    
    gr00t_stat_values_action = {}
    for key, stat_val in rlinf_metadata.statistics.action.items():
        gr00t_stat_values_action[key] = Gr00tDatasetStatisticalValues(
            max=stat_val.max,
            min=stat_val.min,
            mean=stat_val.mean,
            std=stat_val.std,
            q01=stat_val.q01,
            q99=stat_val.q99,
        )
    
    # Convert statistics
    gr00t_statistics = Gr00tDatasetStatistics(
        state=gr00t_stat_values_state,
        action=gr00t_stat_values_action,
    )
    
    # Convert video metadata
    gr00t_video_metadata = {}
    for key, video_meta in rlinf_metadata.modalities.video.items():
        gr00t_video_metadata[key] = Gr00tVideoMetadata(
            resolution=video_meta.resolution,
            channels=video_meta.channels,
            fps=video_meta.fps,
        )
    
    # Import gr00t RotationType if available
    try:
        from gr00t.data.schema import RotationType as Gr00tRotationType
    except ImportError:
        Gr00tRotationType = None
    
    # Convert state metadata
    gr00t_state_metadata = {}
    for key, state_meta in rlinf_metadata.modalities.state.items():
        # Convert rotation_type if needed
        rotation_type_value = None
        if state_meta.rotation_type is not None:
            if Gr00tRotationType is not None:
                # Try to convert to gr00t RotationType enum
                try:
                    rotation_type_value = Gr00tRotationType(state_meta.rotation_type.value)
                except (ValueError, AttributeError):
                    # If conversion fails, use string value
                    rotation_type_value = state_meta.rotation_type.value
            else:
                rotation_type_value = state_meta.rotation_type.value if hasattr(state_meta.rotation_type, 'value') else str(state_meta.rotation_type)
        
        gr00t_state_metadata[key] = Gr00tStateActionMetadata(
            absolute=state_meta.absolute,
            rotation_type=rotation_type_value,
            shape=state_meta.shape,
            continuous=state_meta.continuous,
        )
    
    # Convert action metadata
    gr00t_action_metadata = {}
    for key, action_meta in rlinf_metadata.modalities.action.items():
        # Convert rotation_type if needed
        rotation_type_value = None
        if action_meta.rotation_type is not None:
            if Gr00tRotationType is not None:
                # Try to convert to gr00t RotationType enum
                try:
                    rotation_type_value = Gr00tRotationType(action_meta.rotation_type.value)
                except (ValueError, AttributeError):
                    # If conversion fails, use string value
                    rotation_type_value = action_meta.rotation_type.value
            else:
                rotation_type_value = action_meta.rotation_type.value if hasattr(action_meta.rotation_type, 'value') else str(action_meta.rotation_type)
        
        gr00t_action_metadata[key] = Gr00tStateActionMetadata(
            absolute=action_meta.absolute,
            rotation_type=rotation_type_value,
            shape=action_meta.shape,
            continuous=action_meta.continuous,
        )
    
    # Convert modalities
    gr00t_modalities = Gr00tDatasetModalities(
        video=gr00t_video_metadata,
        state=gr00t_state_metadata,
        action=gr00t_action_metadata,
    )
    
    # Convert embodiment tag
    # Try to convert to gr00t's EmbodimentTag, fallback to NEW_EMBODIMENT
    try:
        gr00t_embodiment_tag = Gr00tEmbodimentTag(rlinf_metadata.embodiment_tag.value)
    except (ValueError, AttributeError):
        # If the tag doesn't exist in gr00t's enum, use NEW_EMBODIMENT
        gr00t_embodiment_tag = Gr00tEmbodimentTag.NEW_EMBODIMENT
    
    # Create gr00t DatasetMetadata
    gr00t_metadata = Gr00tDatasetMetadata(
        statistics=gr00t_statistics,
        modalities=gr00t_modalities,
        embodiment_tag=gr00t_embodiment_tag,
    )
    
    return gr00t_metadata

