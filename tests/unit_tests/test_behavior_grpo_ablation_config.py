import os
import subprocess
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _compose(config_name: str) -> dict:
    config_dir = (
        Path(__file__).parents[2] / "examples" / "agent" / "behavior_qwen" / "config"
    )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return OmegaConf.to_container(compose(config_name=config_name), resolve=False)


def test_reward_ablation_configs_differ_only_in_declared_treatment():
    control = _compose("behavior_grpo_qwen25_7b_textworld_8gpu_control_v3")
    treatment = _compose("behavior_grpo_qwen25_7b_textworld_8gpu_treatment_v3")

    assert control["reward"]["shaping"].pop("premature_end_penalty") is None
    assert treatment["reward"]["shaping"].pop("premature_end_penalty") == -0.5
    assert (
        control["runner"].pop("experiment_name").endswith("control-v3-eager-20260829")
    )
    assert (
        treatment["runner"]
        .pop("experiment_name")
        .endswith("treatment-v3-eager-20260829")
    )
    assert control == treatment
    assert control["runner"]["max_steps"] == 24
    assert control["runner"]["val_check_interval"] == -1
    assert control["runner"]["save_interval"] == 5
    assert control["rollout"]["enforce_eager"] is True


def test_textworld_launcher_requires_all_pinned_release_variables():
    root = Path(__file__).parents[2]
    launcher = root / "examples" / "agent" / "behavior_qwen" / "run_train.sh"
    config_name = "behavior_grpo_qwen25_7b_textworld_8gpu_control_v3"
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        bddl_root = temp / "bddl"
        asset_root = temp / "assets"
        bbox_path = temp / "native_bbox.json"
        bddl_root.mkdir()
        asset_root.mkdir()
        bbox_path.touch()
        required = {
            "BEHAVIOR_INSTANCE_SOURCES": "2026-v3.9.1",
            "BEHAVIOR_BDDL_DEFINITION_ROOT": str(bddl_root),
            "BEHAVIOR_NATIVE_BBOX_PATH": str(bbox_path),
            "BEHAVIOR_ASSET_SCENE_ROOT": str(asset_root),
        }

        for missing in required:
            environment = os.environ.copy() | required
            environment.pop(missing)
            result = subprocess.run(
                ["bash", str(launcher), config_name],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 2
            assert (
                f"ERROR: {missing} is required for TextWorld training" in result.stderr
            )
            assert "train.py" not in result.stderr
