"""Privileged symbolic expert planner for BEHAVIOR semantic-tool tasks (M2).

Reads a ``BehaviorTask``'s ground BDDL goal atoms and emits a semantic-tool
sequence that achieves them through :class:`SemanticACI`, recording the
trajectory for SFT bootstrap (the SFT->GRPO recipe, proposal §3.6).

Legitimacy (0710 proposal §4): the expert uses privileged info (object_scope +
goal atoms), but success is judged by the REAL BDDL ``goal_status`` -- it cannot
fake success, only drive the same tool interface the policy will use.

Scope (M2): handles the predicates that map to the implemented tools --
``inside`` / ``ontop`` (grasp + place), ``toggled_on`` (toggle), ``open`` and
negated-open i.e. "closed" (open/close). ``nextto`` is approximated by
navigation. Unsupported predicates are recorded and skipped (trajectory marked
incomplete), never silently "achieved".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ------------------------------------------------------------------ #
# BDDL predicate -> ordered semantic-tool subsequence
#   each returns a list of (aci_method_name, kwargs); place preds also declare a
#   (movable, target, PredicateName) cache pair so poses are sampled OFFLINE.
# ------------------------------------------------------------------ #
def _inside(o, t):
    return ([("go_to", {"name": o}), ("grasp", {"name": o}),
             ("go_to", {"name": t}), ("place_inside", {"name": o, "container": t})],
            [(o, t, "Inside")])


def _ontop(o, t):
    return ([("go_to", {"name": o}), ("grasp", {"name": o}),
             ("go_to", {"name": t}), ("place_on", {"name": o, "surface": t})],
            [(o, t, "OnTop")])


def _toggled_on(o, t=None):
    return ([("go_to", {"name": o}), ("toggle_on", {"name": o})], [])


def _open(o, t=None):
    return ([("go_to", {"name": o}), ("open", {"name": o})], [])


def _closed(o, t=None):
    return ([("go_to", {"name": o}), ("close", {"name": o})], [])


def _nextto(o, t):
    # no manipulation predicate for "nextto"; approximate by navigating to target
    return ([("go_to", {"name": t})], [])


PREDICATE_TOOLS = {
    "inside": _inside,
    "ontop": _ontop,
    "toggled_on": _toggled_on,
    "open": _open,
    "nextto": _nextto,
}


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
    unsupported: list = field(default_factory=list)   # predicates we couldn't map
    success: bool = False

    def to_sft_row(self) -> dict:
        """SFT-friendly row (structure parallels spatialcode sft_expert.to_row)."""
        return {
            "activity": self.activity,
            "task": self.task_description,
            "success": self.success,
            "num_steps": len(self.steps),
            "tool_steps": [asdict(s) for s in self.steps],
            "unsupported_predicates": self.unsupported,
        }


def _resolve_bddl(task, bddl_name):
    ent = task.object_scope.get(bddl_name)
    obj = getattr(ent, "wrapped_obj", None) if ent is not None else None
    return obj.name if obj is not None else None


def _atom_pred_args(atom):
    """Return (predicate, [bddl_arg_names], negated) from a bddl HEAD atom.

    body is like ['inside', a, b] or ['not', ['open', a]]."""
    body = list(atom.body)
    negated = False
    if body and body[0] == "not":
        negated = True
        body = list(body[1])
    return body[0], body[1:], negated


def plan(task, option_index: int = 0):
    """Build a (tool_calls, cache_pairs, unsupported) plan from a ground goal option."""
    options = task.ground_goal_state_options
    atoms = options[option_index]
    calls, pairs, unsupported = [], [], []
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
        # "not open" == closed
        key = "closed" if (pred == "open" and negated) else pred
        factory = {"closed": _closed}.get(key) or PREDICATE_TOOLS.get(key)
        if factory is None or (negated and key != "closed"):
            unsupported.append((pred, bddl_args, "negated" if negated else "unsupported-predicate"))
            continue
        sub_calls, sub_pairs = factory(*sim_args)
        calls.extend(sub_calls)
        pairs.extend(sub_pairs)
    return calls, pairs, unsupported


def run_expert(aci, task, task_description: str = "", option_index: int = 0) -> ExpertTrajectory:
    """Drive @aci through the planned tool sequence and record the trajectory."""
    calls, pairs, unsupported = plan(task, option_index)
    if pairs:
        aci.build_pose_cache(pairs)          # OFFLINE placement sampling
    traj = ExpertTrajectory(activity=getattr(task, "activity_name", "?"),
                            task_description=task_description, unsupported=unsupported)
    for tool, args in calls:
        res = getattr(aci, tool)(**args)
        traj.steps.append(ExpertStep(tool=tool, args=args, ok=res.ok, reason=res.reason))
    traj.success = aci.is_success()
    return traj
