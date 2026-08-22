# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Integrated online GRPO for the BEHAVIOR semantic-tool planner.

Same wiring as examples/agent/eqa_qwen/train.py with the BEHAVIOR agent loop and
tool worker substituted in. The current deployment uses one Python 3.10 venv: the
tool worker holds OmniGibson in process, so no env-server pool or HTTP hop is on the
training path. ``env_server.py`` remains a debugging surface only.

    python examples/agent/behavior_qwen/train.py \\
        --config-path config --config-name behavior_grpo_qwen25_7b_1gpu
"""
import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.agents.behavior.behavior_agent_loop import BehaviorAgentLoopWorker
from rlinf.agents.behavior.behavior_tool_worker import BehaviorToolWorker, tool_names
from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor import get_actor_worker
from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.rollout.utils import get_rollout_backend_worker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    agent_obs_mode = str(cfg.agentloop.get("obs_mode", "full"))
    tool_obs_mode = str(cfg.tools.behavior.get("obs_mode", "full"))
    if agent_obs_mode != tool_obs_mode:
        raise ValueError(
            "agentloop.obs_mode and tools.behavior.obs_mode must match: the former "
            f"renders the policy schema ({agent_obs_mode!r}) while the latter "
            f"constructs the environment ({tool_obs_mode!r})"
        )
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    agentloop_placement_strategy = NodePlacementStrategy(
        [
            placement.cluster_node_rank
            for placement in rollout_placement_strategy.get_placement(cluster)
        ]
    )
    assert (
        len(agentloop_placement_strategy._node_ranks)
        == component_placement.rollout_dp_size
    ), "agentloop worker num must equal rollout dp size"
    agentloop_group = BehaviorAgentLoopWorker.create_group(
        cfg, component_placement
    ).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=agentloop_placement_strategy,
    )

    inference_group = None
    if (
        component_placement.placement_mode == PlacementMode.DISAGGREGATED
        and cfg.algorithm.recompute_logprobs
    ):
        inference_group = MegatronInference.create_group(
            cfg, component_placement
        ).launch(
            cluster,
            name=cfg.inference.group_name,
            placement_strategy=component_placement.get_strategy("inference"),
        )

    actor_placement_strategy = component_placement.get_strategy("actor")
    # Megatron keeps the MA actor (loss_scales over group/agent/turn); the FSDP
    # path has no MA analog yet, so it falls back to the plain factory worker.
    actor_worker_cls = (
        MAMegatronActor if cfg.actor.training_backend == "megatron"
        else get_actor_worker(cfg)
    )
    actor_group = actor_worker_cls.create_group(cfg, component_placement).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement_strategy
    )

    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_rl_dataset(cfg, tokenizer)

    # The tool worker now BOOTS OMNIGIBSON IN PROCESS, so it is no longer a thin
    # proxy: it needs a visible GPU for PhysX and holds the scene for the whole run.
    # It shares node 0 (and GPU 0) with SGLang, which is why the 1-GPU config drops
    # rollout.gpu_memory_utilization to 0.5. One rank == one activity, because
    # OmniGibson locks the activity at first boot.
    tool_workers = {
        BehaviorToolWorker.create_group(cfg).launch(
            cluster, name="behavior", placement_strategy=NodePlacementStrategy([0])
        ): ToolWorkerInfo(tool_names=tool_names(agent_obs_mode), has_session=True),
    }

    runner = AgentRunner(
        cfg=cfg,
        placement=component_placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout_group,
        inference=inference_group,
        actor=actor_group,
        reward=None,
        agent_loop=agentloop_group,
        tool_workers=tool_workers,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
