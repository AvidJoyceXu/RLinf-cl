"""Privileged symbolic expert planner for BEHAVIOR semantic-tool tasks (M2).

Reads a ``BehaviorTask``'s ground BDDL goal atoms and emits a semantic-tool
sequence that achieves them through :class:`SemanticACI`, recording the
trajectory for SFT bootstrap (the SFT->GRPO recipe, proposal §3.6).

Legitimacy (0710 proposal §4): the expert uses privileged info (object_scope +
goal atoms), but success is judged by the REAL BDDL ``goal_status`` -- it cannot
fake success, only drive the same tool interface the policy will use.

Predicate coverage (0710 §8.5):
  class A (rearrangement): inside, ontop, toggled_on, open/closed -- with
    open-before-place / close-after ordering for door containers.
  class B1 (state transforms): cooked, covered (spray) / not-covered (uncover),
    contains (fill).
  UNSUPPORTED (recorded, never faked): real (slice/dice/recipe products -> B2),
    on_fire (needs heat source), nextto / touching / attached (class C).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from omnigibson.object_states import Open


# ------------------------------------------------------------------ #
# BDDL predicate -> ordered semantic-tool subsequence + cache pairs.
# ------------------------------------------------------------------ #
def _inside(o, t):
    return ([("go_to", {"name": o}), ("grasp", {"name": o}),
             ("go_to", {"name": t}), ("place_inside", {"name": o, "container": t})],
            [(o, t, "Inside")])


def _ontop(o, t):
    return ([("go_to", {"name": o}), ("grasp", {"name": o}),
             ("go_to", {"name": t}), ("place_on", {"name": o, "surface": t})],
            [(o, t, "OnTop")])


def _toggled_on(o):
    return ([("go_to", {"name": o}), ("toggle_on", {"name": o})], [])


def _open(o):
    return ([("go_to", {"name": o}), ("open", {"name": o})], [])


def _cooked(o):
    return ([("go_to", {"name": o}), ("cook", {"name": o})], [])


def _covered(o, sys):
    return ([("go_to", {"name": o}), ("spray", {"name": o, "system_name": sys})], [])


def _uncovered(o, sys):
    return ([("go_to", {"name": o}), ("uncover", {"name": o, "system_name": sys})], [])


def _contains(o, sys):
    return ([("go_to", {"name": o}), ("fill", {"name": o, "system_name": sys})], [])


@dataclass
class ExpertStep:
    tool: str
    args: dict
    ok: bool
    reason: str


@dataclass
class ExpertTrajectory:
    activity: str
    task_description: str = ""
    steps: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    success: bool = False

    @property
    def num_steps(self):
        return len(self.steps)

    def to_sft_row(self) -> dict:
        return {
            "activity": self.activity,
            "task": self.task_description,
            "success": self.success,
            "num_steps": len(self.steps),
            "tool_steps": [asdict(s) for s in self.steps],
            "unsupported_predicates": self.unsupported,
        }


def _resolve_bddl(task, bddl_name):
    """Resolve a BDDL arg to a sim name: object -> wrapped_obj.name, system ->
    system category name (entity.name)."""
    ent = task.object_scope.get(bddl_name)
    if ent is None:
        return None
    if getattr(ent, "is_system", False):
        try:
            return ent.name
        except Exception:
            return None
    obj = getattr(ent, "wrapped_obj", None)
    return obj.name if obj is not None else None


def _atom_pred_args(atom):
    body = list(atom.body)
    negated = False
    if body and body[0] == "not":
        negated = True
        body = list(body[1])
    return body[0], body[1:], negated


@dataclass
class Plan:
    calls: list          # [(tool, kwargs)]
    pairs: list          # [(movable, target, PredicateName)] to pre-cache
    place_targets: list  # ordered unique inside/ontop targets (candidates to open first)
    close_targets: list  # targets that must end closed ("not open" goals)
    unsupported: list


def plan(task, option_index: int = 0) -> Plan:
    atoms = task.ground_goal_state_options[option_index]
    calls, pairs, place_targets, close_targets, unsupported = [], [], [], [], []
    for atom in atoms:
        try:
            if atom.currently_satisfied:
                continue
        except Exception:
            pass
        pred, bddl_args, negated = _atom_pred_args(atom)
        sim_args = [_resolve_bddl(task, a) for a in bddl_args]
        if any(a is None for a in sim_args):
            unsupported.append((pred, bddl_args, "unresolved-args"))
            continue

        sub, sub_pairs = None, []
        if pred == "inside" and not negated:
            sub, sub_pairs = _inside(*sim_args)
            if sim_args[1] not in place_targets:
                place_targets.append(sim_args[1])
        elif pred == "ontop" and not negated:
            sub, sub_pairs = _ontop(*sim_args)
            if sim_args[1] not in place_targets:
                place_targets.append(sim_args[1])
        elif pred == "toggled_on" and not negated:
            sub, _ = _toggled_on(sim_args[0])
        elif pred == "open" and not negated:
            sub, _ = _open(sim_args[0])
        elif pred == "open" and negated:                 # "not open" == closed
            if sim_args[0] not in close_targets:
                close_targets.append(sim_args[0])
            continue
        elif pred == "cooked" and not negated:
            sub, _ = _cooked(sim_args[0])
        elif pred == "covered" and not negated:
            sub, _ = _covered(*sim_args)
        elif pred == "covered" and negated:
            sub, _ = _uncovered(*sim_args)
        elif pred == "contains" and not negated:
            sub, _ = _contains(*sim_args)
        else:
            unsupported.append((pred, bddl_args, "negated" if negated else "unsupported"))
            continue
        calls.extend(sub)
        pairs.extend(sub_pairs)
    return Plan(calls, pairs, place_targets, close_targets, unsupported)


def run_expert(aci, task, task_description: str = "", option_index: int = 0) -> ExpertTrajectory:
    """Drive @aci through the planned tool sequence and record the trajectory.

    Order: open door-containers -> all placements/transforms -> close containers
    that the goal requires closed (class-A open-before-place / close-after)."""
    p = plan(task, option_index)
    traj = ExpertTrajectory(activity=getattr(task, "activity_name", "?"),
                            task_description=task_description, unsupported=p.unsupported)

    def run(steps):
        for tool, args in steps:
            res = getattr(aci, tool)(**args)
            traj.steps.append(ExpertStep(tool=tool, args=args, ok=res.ok, reason=res.reason))

    # 1) OPEN openable place-target containers FIRST -- build_pose_cache seats
    #    objects inside them, which fails if the door is still shut.
    open_steps = []
    for t in p.place_targets:
        obj = aci._resolve(t)
        if obj is not None and Open in getattr(obj, "states", {}):
            open_steps += [("go_to", {"name": t}), ("open", {"name": t})]
    run(open_steps)
    # 2) build the placement cache now that containers are open
    if p.pairs:
        aci.build_pose_cache(p.pairs)
    # 3) placements / transforms
    run(p.calls)
    # 4) close containers the goal requires closed (after filling)
    close_steps = []
    for t in p.close_targets:
        close_steps += [("go_to", {"name": t}), ("close", {"name": t})]
    run(close_steps)

    traj.success = aci.is_success()
    return traj
