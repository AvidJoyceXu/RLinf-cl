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

from __future__ import annotations

from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
)


class RGBWrapper(EnvironmentWrapper):
    """Validate an environment booted with the official RGB sensor config.

    Sensor modalities and render-product sizes must be configured before the
    environment is constructed. Mutating them here used to work with older Kit
    releases, but detaching active Replicator annotators can invalidate the
    OmniGraph on Isaac Sim 5.1.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        robot = env.robots[0]
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(
                sensor, "image_width"
            ):
                continue
            if "zed_link:Camera:0" in sensor_name:
                expected = HEAD_RESOLUTION
            elif "realsense_link:Camera:0" in sensor_name:
                expected = WRIST_RESOLUTION
            else:
                continue
            actual = (sensor.image_height, sensor.image_width)
            if actual != expected:
                raise RuntimeError(
                    f"sensor {sensor_name!r} booted at {actual}, expected {expected}; "
                    "set omni_config.robots[0].sensor_config before construction"
                )
        env.load_observation_space()
