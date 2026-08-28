"""Fail-closed capability matrix for BEHAVIOR rollout harnesses.

Tool schemas describe the policy action space; they do not prove that a backend
implements those methods or that image tensors reach the model. Keep those separate
claims explicit so an RL launch cannot turn a schema-only branch into a reported
multimodal experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessCapabilities:
    backend: str
    obs_modes: frozenset[str]
    policy_rgb_transport: bool
    trajectory_rgb_debug: bool


CAPABILITIES = {
    "textworld": HarnessCapabilities(
        backend="textworld",
        obs_modes=frozenset(
            {
                "full",
                "partial",
                "object",
                "fov",
                "fov_distract",
                "detect",
                "detect_scope",
                "detect_scope_scan",
                "detect_scope_memory",
            }
        ),
        policy_rgb_transport=False,
        trajectory_rgb_debug=False,
    ),
    "omnigibson": HarnessCapabilities(
        backend="omnigibson",
        # fov includes real base camera actions after SemanticACI camera support.
        # Detect remains the simulator-free analytic interface and is not silently
        # advertised on the physical backend.
        obs_modes=frozenset({"full", "partial", "fov"}),
        policy_rgb_transport=False,
        trajectory_rgb_debug=True,
    ),
}


def validate_harness(
    backend: str,
    obs_mode: str,
    *,
    policy_rgb: bool = False,
) -> HarnessCapabilities:
    """Return capabilities or raise before a worker/model/simulator is launched."""
    if backend not in CAPABILITIES:
        raise ValueError(f"unknown BEHAVIOR backend={backend!r}")
    caps = CAPABILITIES[backend]
    if obs_mode not in caps.obs_modes:
        raise ValueError(
            f"obs_mode={obs_mode!r} is not implemented by backend={backend!r}; "
            f"supported={sorted(caps.obs_modes)}. Schema registration alone is not "
            "backend support. Use backend='textworld' for analytic detect modes."
        )
    if policy_rgb and not caps.policy_rgb_transport:
        raise ValueError(
            "policy RGB transport is not implemented: BehaviorAgentLoopWorker "
            "currently tokenizes JSON tool text only. Debug trajectory video is a "
            "separate recorder and must not be reported as multimodal policy input."
        )
    return caps


__all__ = ["CAPABILITIES", "HarnessCapabilities", "validate_harness"]
