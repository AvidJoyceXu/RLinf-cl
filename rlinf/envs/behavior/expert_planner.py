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
  class B2 (product-creating): real(half__X) via slice; real(diced__X) via dice;
    a diced system that also has a contains(container, system) goal is diced
    INSIDE that container so both atoms fall out of the real dicing act.
  UNSUPPORTED (recorded, never faked): real(recipe product) e.g. pizza (needs
    RecipeRule), on_fire (needs heat source), nextto / touching / attached (C).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from omnigibson.object_states import Open


# ------------------------------------------------------------------ #
# BDDL predicate -> ordered semantic-tool subsequence + cache pairs.
# ------------------------------------------------------------------ #
def _inside(o, t):
    return (
        [
            ("go_to", {"name": o}),
            ("grasp", {"name": o}),
            ("go_to", {"name": t}),
            ("place_inside", {"name": o, "container": t}),
        ],
        [(o, t, "Inside")],
    )


def _ontop(o, t):
    return (
        [
            ("go_to", {"name": o}),
            ("grasp", {"name": o}),
            ("go_to", {"name": t}),
            ("place_on", {"name": o, "surface": t}),
        ],
        [(o, t, "OnTop")],
    )


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


def _scope_object(entity):
    """Return the concrete object across old BDDL wrappers and v3.9 scopes.

    Older runtimes store an entity wrapper with ``wrapped_obj``. BEHAVIOR v3.9
    stores the concrete OmniGibson object directly (the instance loader calls its
    ``load_state`` method), so requiring ``wrapped_obj`` makes every valid 2026
    goal argument look unresolved.
    """
    wrapped = getattr(entity, "wrapped_obj", None)
    return wrapped if wrapped is not None else entity


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
    obj = _scope_object(ent)
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
    calls: list  # [(tool, kwargs)]
    pairs: list  # [(movable, target, PredicateName)] to pre-cache
    place_targets: (
        list  # ordered unique inside/ontop targets (candidates to open first)
    )
    close_targets: list  # targets that must end closed ("not open" goals)
    unsupported: list
    open_targets: list = field(default_factory=list)  # targets a goal wants OPEN


def _synset_cat(bddl_inst: str) -> str:
    """'half__log.n.01_1' -> 'half__log' (strip the '.n.NN_k' instance suffix)."""
    return bddl_inst.rsplit(".n.", 1)[0]


def _product_transform(prod_inst: str):
    """From a `real(...)` product instance derive (transform_tool, source_cat).

    Slicing yields object parts named ``half__<cat>``; dicing yields a
    ``diced__<cat>`` (optionally ``cooked__diced__<cat>``) particle system.
    Returns (None, None) for products of other rules (e.g. recipe outputs)."""
    cat = _synset_cat(prod_inst)
    if cat.startswith("half__"):
        return "slice", cat[len("half__") :]
    for pre in ("cooked__diced__", "diced__"):
        if cat.startswith(pre):
            return "dice", cat[len(pre) :]
    return None, None


def _sources_of(task, source_cat: str):
    """Existing scope objects whose synset category == @source_cat, as
    [(bddl_inst, sim_name)] -- the whole objects a slice/dice consumes."""
    out = []
    for inst, ent in task.object_scope.items():
        if getattr(ent, "is_system", False) or _synset_cat(inst) != source_cat:
            continue
        obj = _scope_object(ent)
        if obj is not None:
            out.append((inst, obj.name))
    return out


def plan(task, option_index: int = 0) -> Plan:
    atoms = task.ground_goal_state_options[option_index]
    calls, pairs, place_targets, close_targets, unsupported = [], [], [], [], []
    open_targets = []  # containers a positive open(...) goal wants left open
    real_products = []  # product bddl_inst that must be created (slice/dice)
    contains_atoms = []  # (container_sim, system_sim, system_bddl_inst)
    for atom in atoms:
        try:
            if atom.currently_satisfied:
                continue
        except Exception:
            pass
        pred, bddl_args, negated = _atom_pred_args(atom)

        # `real(product)` names a not-yet-existing product; resolve its SOURCE,
        # not the (unresolvable) product arg. Deferred to the post-pass below.
        if pred == "real" and not negated:
            real_products.append(bddl_args[0])
            continue

        sim_args = [_resolve_bddl(task, a) for a in bddl_args]

        # `contains(container, system)` for a not-yet-real diced/melted system is
        # best satisfied AS A SIDE EFFECT of dicing the source inside the
        # container. Defer; the post-pass couples it or falls back to `fill`.
        if pred == "contains" and not negated:
            contains_atoms.append((sim_args[0], bddl_args[1]))
            continue

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
            open_targets.append(sim_args[0])
        elif pred == "open" and negated:  # "not open" == closed
            if sim_args[0] not in close_targets:
                close_targets.append(sim_args[0])
            continue
        elif pred == "cooked" and not negated:
            sub, _ = _cooked(sim_args[0])
        elif pred == "covered" and not negated:
            sub, _ = _covered(*sim_args)
        elif pred == "covered" and negated:
            sub, _ = _uncovered(*sim_args)
        else:
            unsupported.append(
                (pred, bddl_args, "negated" if negated else "unsupported")
            )
            continue
        calls.extend(sub)
        pairs.extend(sub_pairs)

    # ---- post-pass: resolve real(...) products via slice/dice ----
    # container that must end up containing each diced system (couples with dice)
    contains_by_sys = {
        _synset_cat(sys_inst): cont
        for (cont, sys_inst) in contains_atoms
        if cont is not None
    }
    handled_sys, sliced = set(), set()
    for prod_inst in real_products:
        transform, src_cat = _product_transform(prod_inst)
        if transform is None:
            unsupported.append(("real", [prod_inst], "non-slice/dice product"))
            continue
        sources = _sources_of(task, src_cat)
        if not sources:
            unsupported.append(("real", [prod_inst], f"no source {src_cat} in scope"))
            continue
        prod_cat = _synset_cat(prod_inst)
        if transform == "slice":
            for _, sname in sources:  # slice every whole source once
                if sname in sliced:
                    continue
                sliced.add(sname)
                calls += [("go_to", {"name": sname}), ("slice", {"name": sname})]
        else:  # dice -- optionally routed through a container it must fill
            container = contains_by_sys.get(prod_cat)
            for _, sname in sources:
                if sname in sliced:
                    continue
                sliced.add(sname)
                if container is not None:
                    pairs.append((sname, container, "Inside"))
                    if container not in place_targets:
                        place_targets.append(container)
                    calls += [
                        ("go_to", {"name": sname}),
                        ("grasp", {"name": sname}),
                        ("go_to", {"name": container}),
                        ("place_inside", {"name": sname, "container": container}),
                        ("go_to", {"name": sname}),
                        ("dice", {"name": sname}),
                    ]
                    handled_sys.add(prod_cat)
                else:
                    calls += [("go_to", {"name": sname}), ("dice", {"name": sname})]

    # leftover contains(container, system) not produced by dicing -> fill directly
    for cont, sys_inst in contains_atoms:
        if _synset_cat(sys_inst) in handled_sys:
            continue
        sys_sim = _resolve_bddl(task, sys_inst)
        if cont is None or sys_sim is None:
            unsupported.append(("contains", [sys_inst], "unresolved / no producer"))
            continue
        calls += _contains(cont, sys_sim)[0]

    return Plan(calls, pairs, place_targets, close_targets, unsupported, open_targets)


def run_expert(
    aci, task, task_description: str = "", option_index: int = 0
) -> ExpertTrajectory:
    """Drive @aci through the planned tool sequence and record the trajectory.

    Order: open door-containers -> all placements/transforms -> close containers
    that the goal requires closed (class-A open-before-place / close-after)."""
    p = plan(task, option_index)
    traj = ExpertTrajectory(
        activity=getattr(task, "activity_name", "?"),
        task_description=task_description,
        unsupported=p.unsupported,
    )

    def run(steps):
        for tool, args in steps:
            res = getattr(aci, tool)(**args)
            traj.steps.append(
                ExpertStep(tool=tool, args=args, ok=res.ok, reason=res.reason)
            )

    # 1) OPEN openable place-target containers FIRST -- build_pose_cache seats
    #    objects inside them, which fails if the door is still shut.
    opened = []
    open_steps = []
    for t in p.place_targets:
        obj = aci._resolve(t)
        if obj is not None and Open in getattr(obj, "states", {}):
            open_steps += [("go_to", {"name": t}), ("open", {"name": t})]
            opened.append(t)
    run(open_steps)
    # 2) build the placement cache now that containers are open
    if p.pairs:
        aci.build_pose_cache(p.pairs)
    # 3) placements / transforms
    run(p.calls)
    # 4) close containers that must end closed. This is close_targets ("not open"
    #    goals, e.g. the car) PLUS every container WE opened to place into -- the
    #    latter is essential because a "not open(fridge)" goal that started
    #    satisfied is filtered out of close_targets, yet we just opened that
    #    fridge. Skip anything a positive open(...) goal wants left open.
    close_list = list(p.close_targets)
    for t in opened:
        if t not in close_list and t not in p.open_targets:
            close_list.append(t)
    close_steps = []
    for t in close_list:
        close_steps += [("go_to", {"name": t}), ("close", {"name": t})]
    run(close_steps)

    traj.success = aci.is_success()
    return traj
