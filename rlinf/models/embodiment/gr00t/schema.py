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
Schema definitions for RLinf GR00T integration.
These classes are compatible with n1d6 data format and provide interfaces
compatible with the old gr00t transform system.
"""

from enum import Enum
from typing import Optional

import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel, Field, field_serializer

from rlinf.models.embodiment.gr00t.embodiment_tags import EmbodimentTag


class ModalityConfig(BaseModel):
    """Configuration for a modality defining how data should be sampled and loaded.

    This class is compatible with n1d6 ModalityConfig structure.
    """

    delta_indices: list[int] = Field(
        ...,
        description="Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices.",
    )
    modality_keys: list[str] = Field(
        ..., description="The keys to load for the modality in the dataset."
    )
    sin_cos_embedding_keys: list[str] | None = Field(
        default=None,
        description="Optional list of keys to apply sin/cos encoding. If None or empty, use min/max normalization for all keys.",
    )
    mean_std_embedding_keys: list[str] | None = Field(
        default=None,
        description="Optional list of keys to apply mean/std normalization. If None or empty, use min/max normalization for all keys.",
    )


class RotationType(Enum):
    """Type of rotation representation"""

    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    EULER_ANGLES_RPY = "euler_angles_rpy"
    EULER_ANGLES_RYP = "euler_angles_ryp"
    EULER_ANGLES_PRY = "euler_angles_pry"
    EULER_ANGLES_PYR = "euler_angles_pyr"
    EULER_ANGLES_YRP = "euler_angles_yrp"
    EULER_ANGLES_YPR = "euler_angles_ypr"


class DatasetStatisticalValues(BaseModel):
    """Statistical values for a dataset feature."""

    max: NDArray = Field(..., description="Maximum values")
    min: NDArray = Field(..., description="Minimum values")
    mean: NDArray = Field(..., description="Mean values")
    std: NDArray = Field(..., description="Standard deviation")
    q01: NDArray = Field(..., description="1st percentile values")
    q99: NDArray = Field(..., description="99th percentile values")

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: NDArray) -> list[float]:
        return v.tolist()  # type: ignore

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetStatisticalValues":
        """Create from dictionary with list values."""
        return cls(
            max=np.array(data["max"]),
            min=np.array(data["min"]),
            mean=np.array(data["mean"]),
            std=np.array(data["std"]),
            q01=np.array(data["q01"]),
            q99=np.array(data["q99"]),
        )


class DatasetStatistics(BaseModel):
    """Statistics for state and action modalities."""

    state: dict[str, DatasetStatisticalValues] = Field(
        ..., description="Statistics of the state"
    )
    action: dict[str, DatasetStatisticalValues] = Field(
        ..., description="Statistics of the action"
    )


class VideoMetadata(BaseModel):
    """Metadata of the video modality"""

    resolution: tuple[int, int] = Field(..., description="Resolution of the video")
    channels: int = Field(..., description="Number of channels in the video", gt=0)
    fps: float = Field(..., description="Frames per second", gt=0)


class StateActionMetadata(BaseModel):
    """Metadata for state or action modality."""

    absolute: bool = Field(..., description="Whether the state or action is absolute")
    rotation_type: Optional[RotationType] = Field(
        None, description="Type of rotation, if any"
    )
    shape: tuple[int, ...] = Field(..., description="Shape of the state or action")
    continuous: bool = Field(..., description="Whether the state or action is continuous")


class DatasetModalities(BaseModel):
    """Metadata of the modalities."""

    video: dict[str, VideoMetadata] = Field(..., description="Metadata of the video")
    state: dict[str, StateActionMetadata] = Field(..., description="Metadata of the state")
    action: dict[str, StateActionMetadata] = Field(..., description="Metadata of the action")


class DatasetMetadata(BaseModel):
    """Metadata of the trainable dataset.

    This class is compatible with n1d6 dataset_statistics.json format.
    The format is: {embodiment_tag: {modality: {key: {stat_type: values}}}}
    """

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(..., description="Embodiment tag of the dataset")

    @classmethod
    def from_n1d6_format(
        cls,
        n1d6_stats: dict,
        embodiment_tag: EmbodimentTag,
        modality_config: dict,
    ) -> "DatasetMetadata":
        """
        Create DatasetMetadata from n1d6 format dataset_statistics.json.

        Args:
            n1d6_stats: Dictionary in format {modality: {key: {stat_type: values}}}
            embodiment_tag: The embodiment tag
            modality_config: Modality configuration dict with video/state/action keys

        Returns:
            DatasetMetadata instance
        """
        # Extract statistics
        statistics = DatasetStatistics(
            state={
                key: DatasetStatisticalValues.from_dict(stats)
                for key, stats in n1d6_stats.get("state", {}).items()
            },
            action={
                key: DatasetStatisticalValues.from_dict(stats)
                for key, stats in n1d6_stats.get("action", {}).items()
            },
        )

        # Build modalities metadata
        # For video, we need to infer from modality_config or use defaults
        video_modalities = {}
        if "video" in modality_config:
            video_config = modality_config["video"]
            # Handle both ModalityConfig objects and dicts
            if hasattr(video_config, "modality_keys"):
                video_keys = video_config.modality_keys
            else:
                video_keys = video_config.get("modality_keys", [])
            
            for key in video_keys:
                # Remove "video." prefix if present
                clean_key = key.replace("video.", "") if key.startswith("video.") else key
                # Default video metadata - use input resolution (256x256) instead of target resolution (224x224)
                # The transform pipeline will resize to target resolution (224x224) later
                # This allows VideoToTensor to accept the actual input resolution
                video_modalities[clean_key] = VideoMetadata(
                    resolution=(256, 256),  # Input resolution from LIBERO environment
                    channels=3,
                    fps=30.0,
                )

        # For state and action, infer shape from statistics
        state_modalities = {}
        for key, stats in n1d6_stats.get("state", {}).items():
            # Infer shape from mean array length
            mean_array = stats.get("mean", [])
            if isinstance(mean_array, list):
                shape = (len(mean_array),)
            else:
                shape = tuple(mean_array.shape) if hasattr(mean_array, "shape") else (len(mean_array),)
            state_modalities[key] = StateActionMetadata(
                absolute=True,  # Default assumption
                rotation_type=None,
                shape=shape,
                continuous=True,
            )

        action_modalities = {}
        for key, stats in n1d6_stats.get("action", {}).items():
            # Infer shape from mean array length
            mean_array = stats.get("mean", [])
            if isinstance(mean_array, list):
                shape = (len(mean_array),)
            else:
                shape = tuple(mean_array.shape) if hasattr(mean_array, "shape") else (len(mean_array),)
            action_modalities[key] = StateActionMetadata(
                absolute=True,  # Default assumption
                rotation_type=None,
                shape=shape,
                continuous=True,
            )

        modalities = DatasetModalities(
            video=video_modalities,
            state=state_modalities,
            action=action_modalities,
        )

        return cls(
            statistics=statistics,
            modalities=modalities,
            embodiment_tag=embodiment_tag,
        )

