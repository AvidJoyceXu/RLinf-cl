"""BEHAVIOR-TextWorld env -- a drop-in replacement for ``env_server.BehaviorEnv``.

Same constructor keywords, same ``start`` / ``call`` / ``end`` / ``close`` methods,
same response bodies. ``BehaviorToolWorker`` picks between the two with
``tools.behavior.backend: omnigibson | textworld`` and needs no other change; the
agent loop, the prompt builder, the 17 tool schemas and the reward module do not
know which one they are talking to.

WHAT CHANGES, AND WHY IT IS WORTH IT:

  * ``start`` is ~1 ms, not ~0.5-1.3 s, and a tool call is ~50 us. There is no Kit,
    no USD, no GPU, no boot.
  * OmniGibson's two structural constraints are gone. **One activity per process**
    came from Kit locking ``HEADLESS`` at first boot; there is no Kit, so one worker
    can host every activity (``env_for(activity)`` caches one world each). **One
    session at a time** came from there being a single mutable scene; here each
    session gets its own ``SymbolicWorld``, so sessions are independent by
    construction. ``BehaviorToolWorker`` still serialises them behind its existing
    lock -- lifting that is a worker change, not an env one.
  * Every (tool, object) pair has a defined outcome, which was the point: the
    geometric ``_place`` refuses any pair outside the expert's pose cache, so
    off-plan exploration could not be scored. Nothing here is uncached.

WHAT IS LOST, and these are the reasons this is a *second* benchmark rather than a
replacement for the geometric one: no geometry (see the fidelity gaps in
``symbolic_world``), and **no instance variety** -- every BEHAVIOR activity ships
exactly one BDDL definition (measured: 1016/1016 have ``get_instance_count == 1``),
so ``instance_id`` selects nothing and the start state is deterministic per
activity. On the OmniGibson path a GRPO group's within-group variance came partly
from different sampled instances; here it comes from sampling temperature alone.
Breadth replaces depth: 703 verified-solvable activities against 50 instance-backed
ones.
"""
from __future__ import annotations

from uuid import uuid4

from rlinf.envs.behavior.nl_goal import render_goal_nl
from rlinf.envs.behavior.rft_sample import result_payload
from rlinf.envs.behavior.sft_build import goal_atoms_to_lines
from rlinf.envs.behavior.symbolic_world import SymbolicACI, SymbolicWorld


class _Atom:
    """Adapter giving a ground-atom list the ``.body`` attribute
    ``sft_build.goal_atoms_to_lines`` reads, so the goal block is rendered by the
    same function on both backends instead of a near-copy that could drift."""

    __slots__ = ("body",)

    def __init__(self, body):
        self.body = body


class BehaviorTextWorld:
    """One activity's symbolic world, behind the ``BehaviorEnv`` interface."""

    def __init__(self, activity: str, obs_mode: str = "full",
                 instances_per_activity: int = 0, rgb: bool = False,
                 partial_scene: bool = True, fast_reset: bool = False):
        # rgb / partial_scene / fast_reset are accepted and ignored: they are all
        # geometry-loading knobs, and there is no geometry. Kept in the signature so
        # one config block drives either backend, rather than silently rejecting
        # keys the OmniGibson path needs.
        self.activity = activity
        self.obs_mode = obs_mode
        self.scene = None                  # no scene is loaded; do not invent a name

        self.world = SymbolicWorld(activity)
        self.aci = SymbolicACI(self.world, obs_mode=obs_mode)
        self.goal_lines = goal_atoms_to_lines(
            [_Atom(a) for a in self.world.ground_goal_atoms()])
        self.goal_nl, self.goal_nl_fallbacks = render_goal_nl(activity)

        # BDDL ships one definition per activity, so there is exactly one instance.
        # The list is kept (rather than dropped) because build_rl_dataset writes an
        # instance_id into every RL row and `start` must be able to accept it.
        self.instances = [0]
        self.session_id: str | None = None

    # ------------------------------------------------------------------ #
    def start(self, instance_id: int | None = None) -> dict:
        """Reset to the activity's canonical start state and open a session.

        Rebuilds the world from the BDDL initial conditions rather than undoing the
        previous episode's effects: it costs about a millisecond and cannot leave
        residue, which is the failure mode that made the OmniGibson reset both slow
        and delicate.
        """
        if instance_id not in (None, 0):
            # Loud, not silent: a dataset row asking for instance 322 is asking for
            # geometric variety this backend does not have, and quietly serving the
            # only instance would make the run look like it covered more than it did.
            raise KeyError(
                f"instance {instance_id} not available for {self.activity}: "
                f"BEHAVIOR-TextWorld has exactly one instance per activity (BDDL "
                f"ships one definition each). Build the RL dataset with "
                f"`build_rl_dataset --backend textworld`, which emits instance 0."
            )
        self.world.reset()
        self.aci = SymbolicACI(self.world, obs_mode=self.obs_mode)
        self.session_id = uuid4().hex
        return {
            "session_id": self.session_id,
            "activity": self.activity,
            "scene": self.scene,
            "instance_id": 0,
            "obs_mode": self.obs_mode,
            "goal_lines": self.goal_lines,
            "goal_nl": self.goal_nl,
            "goal_nl_fallbacks": self.goal_nl_fallbacks,
            # Same guard as the OmniGibson path: a goal already true at reset would
            # hand the policy a free success. `audit()` excludes these activities up
            # front, so this should never fire -- which is exactly why it is checked.
            "contaminated": bool(self.world.is_success()),
        }

    def call(self, name: str, arguments: dict) -> dict:
        """Dispatch one tool; identical ``{payload, meta, terminal}`` contract."""
        fn = getattr(self.aci, name, None)
        if fn is None or name.startswith("_"):
            return {"payload": {"ok": False, "reason": f"unknown_tool:{name}"},
                    "meta": self._meta(), "terminal": False}
        try:
            res = fn(**(arguments or {}))
        except TypeError as ex:                       # bad/missing arguments
            return {"payload": {"ok": False, "reason": f"bad_arguments: {ex}"},
                    "meta": self._meta(), "terminal": False}
        return {"payload": result_payload(name, res),
                "meta": self._meta(),
                "terminal": name == "end_task"}

    def _meta(self) -> dict:
        """Real BDDL goal evaluation over the literal set -- the same evaluator the
        OmniGibson path uses, reading symbolic state instead of geometry."""
        gs = self.world.goal_status()
        n_sat = len(gs["satisfied"])
        n_goal = n_sat + len(gs["unsatisfied"])
        return {"num_satisfied": n_sat, "num_goal": n_goal,
                "is_success": (len(gs["unsatisfied"]) == 0 and n_sat > 0)}

    def end(self) -> dict:
        self.session_id = None
        return {"ok": True}

    def close(self) -> None:
        pass


def _main() -> None:
    """Drive one episode by hand -- the debugging surface ``env_server`` gives the
    OmniGibson backend, minus the HTTP server nobody needs without a boot cost."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activity", default="picking_up_trash")
    ap.add_argument("--obs-mode", default="full",
                    choices=["full", "partial", "object"])
    ap.add_argument("--expert", action="store_true",
                    help="run the symbolic expert plan and report BDDL success")
    args = ap.parse_args()

    env = BehaviorTextWorld(args.activity, obs_mode=args.obs_mode)
    body = env.start()
    print(json.dumps({k: v for k, v in body.items() if k != "session_id"}, indent=1))

    if args.expert:
        from rlinf.envs.behavior.symbolic_expert import solve

        out = solve(args.activity, obs_mode=args.obs_mode)
        for step in out["trace"]:
            print(f"  {'OK ' if step['ok'] else 'REJ'} {step['name']}"
                  f"({step['arguments']}) -> {step['reason'][:70]}")
        print(f"success={out['success']} steps={out['n_steps']} "
              f"rejected={out['n_rejected']}")
    else:
        print(json.dumps(env.call("observe", {}), indent=1))


if __name__ == "__main__":
    _main()
