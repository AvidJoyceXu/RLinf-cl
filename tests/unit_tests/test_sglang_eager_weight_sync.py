from types import SimpleNamespace

from rlinf.hybrid_engines.sglang.common.io_struct import (
    SyncHFWeightInput,
    SyncHFWeightOutput,
)
from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler


class _Receiver:
    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.calls = []

    def recv(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.state_dict)


def test_eager_scheduler_accepts_one_weight_bucket():
    receiver = _Receiver({"weight": object(), "bucket_length": 1})
    loaded = []
    flushed = []
    scheduler = SimpleNamespace(
        weight_reload="sync",
        cfg=SimpleNamespace(rollout=SimpleNamespace(enforce_eager=True)),
        _rlinf_worker=receiver,
        _actor_group_name="ActorGroup",
        actor_weight_rank=3,
        is_weight_offloaded=False,
        batch_load_hf_weight=lambda state: loaded.append(state),
        weight_norm_dict=None,
        flush_cache=lambda: flushed.append(True),
    )

    result = Scheduler.sync_hf_weight(scheduler, SyncHFWeightInput())

    assert isinstance(result, SyncHFWeightOutput)
    assert receiver.calls == [{"src_group_name": "ActorGroup", "src_rank": 3}]
    assert loaded == [{"weight": scheduler._rlinf_worker.state_dict["weight"]}]
    assert flushed == [True]
