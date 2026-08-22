"""BEHAVIOR-TextWorld -- the symbolic-layer dynamics ground truth.

The geometric ACI (``semantic_tools.SemanticACI``) has ground truth for exactly two
things: object poses at scene reset (offline Monte-Carlo sampling) and the pose
changes along an *expert* trajectory (``build_pose_cache`` walks the expert's
(movable, target, predicate) triples). It has none for an arbitrary tool applied to
an arbitrary object -- ``_place`` refuses any pair outside the cache, which confines
the policy to the expert's plan and makes off-plan exploration unscoreable. That is
a kinematics problem, and it is the only thing blocking RL from running on more than
a handful of activities.

This module removes kinematics from the loop instead of solving it. The state is a
set of BDDL ground literals; the 17 tools are precondition/effect operators over
that set; nothing is sampled, nothing is stepped, and every (tool, object) pair has
a defined outcome. One episode costs microseconds and needs no GPU, no Kit, and no
OmniGibson import -- this file deliberately imports **only bddl**.

WHAT IS BORROWED AND WHAT IS OURS (the honesty boundary):

  * borrowed, unmodified: ``bddl.trivial_backend.TrivialSimulator`` holds the
    literals and its own entailments (filled=>contains, ontop=>nextto);
    ``bddl.activity.get_goal_conditions`` / ``evaluate_goal_conditions`` compile and
    evaluate the goal, quantifiers and all. **Reward still comes from the real BDDL
    evaluator over real state** -- exactly the red line the OmniGibson path holds
    (0710 §4). We never score our own dynamics.
  * ours: the transition function below. That is the approximation, and it is the
    thing to be sceptical of. It is stated as explicit preconditions and effects so
    it can be read and disagreed with, rather than hidden inside a physics engine.

PRECONDITIONS COME FROM BDDL'S OWN ANNOTATIONS, not from invention. ``openable``,
``toggleable``, ``cookable``, ``sliceable``, ``diceable``, ``fillable``,
``substance`` and ``sceneObject`` are read from
``bddl/generated_data/propagated_annots_canonical.json`` -- the same per-synset
property table OmniGibson derives its object states from. So ``cook(floor_1)`` is
refused because ``floor.n.01`` is not annotated ``cookable``, not because we decided
floors are not food.

COVERAGE, MEASURED (``python -m rlinf.envs.behavior.symbolic_world --audit``):
922 of 1016 activities have every goal predicate in the affectable set
{ontop, inside, open, toggled_on, cooked, covered, real}. The other 94 need
folded / draped / on_fire / frozen / saturated / under / broken / unfolded, which no
tool in this ACI produces. They are **excluded from the benchmark, not faked**.

KNOWN FIDELITY GAPS vs the geometric layer -- read these before quoting a number:

  1. No geometry at all. Containers have unbounded capacity; a placement that would
     not physically fit succeeds here. The geometric layer's own answer to this was
     "refuse anything uncached", which is not obviously the better error.
  2. Proximity is relational, not metric: you are near what you navigated to, near
     what rests on/in it, and near what it rests on/in. A whole floor's contents
     therefore become reachable at once, where the sim used a 1.5 m radius.
  3. Observability is room-scoped or container-scoped, never a view cone (see
     ``SymbolicACI._visible``). ``partial`` is the room gate and is measurably weak:
     87.7% of the 740 solvable activities are single-room. ``object`` adds the honest
     part -- a shut container hides its contents, so finding an object can require
     searching. It bites on 30.3% of activities (224/740 have >=1 object inside a
     closed container at reset; 1085 of 7490 objects, 14.5%).
  4. ``uncover`` needs no cleaning tool, mirroring ``SemanticACI._set_covered``.
     BEHAVIOR proper requires a ``particleRemover``.
  5. ``cook`` needs no heat source, again mirroring the geometric ACI.

Gaps 4 and 5 are deliberate *parity* with the geometric layer rather than
improvements, so the two backends stay comparable and SFT data transfers. Gaps 1-3
are where the symbolic layer is genuinely a different task.
"""
from __future__ import annotations

import dataclasses
import functools
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import bddl
from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_object_scope,
)
from bddl.backend_abc import BDDLBackend
from bddl.logic_base import BinaryAtomicFormula
from bddl.trivial_backend import (
    TrivialBackend,
    TrivialGenericObject,
    TrivialSimulator,
)

# Goal predicates at least one of the 17 tools can drive. Anything outside this set
# makes an activity unsolvable in the symbolic layer, and `audit()` excludes it
# rather than letting the policy grind against an impossible goal.
AFFECTABLE_PREDICATES = frozenset(
    {"ontop", "inside", "open", "closed", "toggled_on", "cooked", "covered",
     "real", "future", "filled", "contains", "nextto", "grasped"}
)

# Relations invalidated when an object is picked up, sliced, or re-placed. `nextto`
# and `contains` are omitted on purpose: TrivialSimulator maintains those itself as
# entailments of ontop/filled, and clearing them here would fight its bookkeeping.
_POSITIONAL = ("ontop", "inside", "under", "touching", "overlaid", "draped")


# --------------------------------------------------------------------------- #
# sim-free property annotations
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _property_table() -> dict:
    definition_root = os.environ.get("BEHAVIOR_BDDL_DEFINITION_ROOT")
    package_root = os.path.dirname(definition_root) if definition_root else os.path.dirname(
        bddl.__file__
    )
    path = os.path.join(
        package_root, "generated_data", "propagated_annots_canonical.json"
    )
    with open(path) as f:
        return json.load(f)


def synset_of(scope_name: str) -> str:
    """``can__of__soda.n.01_1`` -> ``can__of__soda.n.01``."""
    return scope_name.rsplit("_", 1)[0]


def properties_of(scope_name: str) -> set:
    return set(_property_table().get(synset_of(scope_name), {}))


def _word(synset: str) -> str:
    """``can__of__soda.n.01`` -> ``can__of__soda`` (the readable category)."""
    return synset.split(".n.")[0]


def lemma_of(scope_name: str) -> str:
    """``diced__bell_pepper.n.01_1`` -> ``diced__bell_pepper``.

    The sense-number-free identity. Transform products do not inherit their source's
    WordNet sense (``bell_pepper.n.02`` dices into ``diced__bell_pepper.n.01``), so
    anything relating a source to its product must compare lemmas, not synsets.
    """
    return _word(synset_of(scope_name))


# Prefix -> the tool that produces it. `dice` runs the slice stage itself, so a
# `diced__` product is reachable from the whole object either way.
_TRANSFORM_PREFIXES = (("cooked__diced__", "dice"), ("diced__", "dice"),
                       ("half__", "slice"))


def producer_of(world, product: str):
    """``(whole_object, tool)`` that would bring @product into existence, or None.

    Lives here rather than in ``symbolic_expert`` because the ACI needs it too: a
    refusal that says an object does not exist yet without saying what makes it exist
    is the one refusal shape this environment measured a policy looping on (five of
    the five worst repeat-loops in the 164-episode run were exactly that message).
    """
    lemma = lemma_of(product)
    for prefix, tool in _TRANSFORM_PREFIXES:
        if lemma.startswith(prefix):
            base = lemma[len(prefix):]
            break
    else:
        return None
    for name in world.scope_names:
        if lemma_of(name) == base and world.is_real(name):
            return name, tool
    return None


def display_names(scope_names) -> tuple[dict, dict]:
    """Map BDDL scope names to the shorter names the policy sees, and back.

    ``can__of__soda.n.01_1`` -> ``can__of__soda_1``. The point is transfer: the SFT
    cold start was trained on OmniGibson registry names (``ashcan_1``), so dropping
    the WordNet suffix keeps the prompt distribution close. Two synsets sharing a
    lemma (``can.n.01`` / ``can.n.03``) would collide, and there the FULL scope name
    is used for both -- an uglier name beats an ambiguous one.
    """
    lemma_to_synsets: dict[str, set] = {}
    for name in scope_names:
        syn = synset_of(name)
        lemma_to_synsets.setdefault(_word(syn), set()).add(syn)

    to_display, from_display = {}, {}
    for name in scope_names:
        syn = synset_of(name)
        lemma = _word(syn)
        idx = name[len(syn) + 1:]
        disp = name if len(lemma_to_synsets[lemma]) > 1 else f"{lemma}_{idx}"
        to_display[name] = disp
        from_display[disp] = name
    return to_display, from_display


# --------------------------------------------------------------------------- #
# quiet backend
# --------------------------------------------------------------------------- #
# bddl 3.7.0 ships eight Trivial*Predicate classes whose STATE_NAME is a copy-paste
# leftover from a neighbouring class. Verified by asking the backend for each of the
# 27 domain predicates and comparing:
#
#     hot -> 'on_fire'     empty    -> 'on_fire'   filled  -> 'ontop'
#     contains -> 'ontop'  nextto   -> 'ontop'     touching-> 'ontop'
#     overlaid -> 'ontop'  attached -> 'ontop'
#
# Their `_evaluate` bodies call the RIGHT getter, so bddl's own evaluation is
# unaffected and this went unnoticed. Two things downstream are not:
#
#   * `get_ground_options` builds `[STATE_NAME, input1, input2]`, so ground goal
#     atoms render `contains(washer, softener)` as `ontop(washer, softener)` -- a
#     wrong goal in the prompt, and one `compile_state` then re-compiles as a
#     genuinely wrong ONTOP check.
#   * subclassing to silence the prints must NOT derive the getter from STATE_NAME,
#     which is what this function did at first. That silently evaluated `contains`
#     as `ontop` -- a reward-corrupting bug of exactly the kind the honesty rules
#     exist to catch, found only because two activities failed to solve.
#
# So the predicate NAME the backend was asked for is the authority, and it overrides
# STATE_NAME rather than being read from it.
@functools.lru_cache(maxsize=None)
def _quiet(base, predicate_name: str):
    """bddl's Trivial*Predicate classes ``print()`` on every ``_evaluate``.

    Goal evaluation runs after every tool call, so unmuted they emit tens of
    thousands of lines per rollout and bury the training log. Subclass, drop the
    print, and pin STATE_NAME to @predicate_name (see the note above). Preferred over
    ``redirect_stdout``, which is process-global and would corrupt output once
    sessions run concurrently -- which, with no simulator to serialise, they can.
    """
    # Read through the SIMULATOR, not the object wrapper. The wrapper is missing
    # `get_filled` entirely (the only one of the 27; bddl's own FilledPredicate
    # papers over it by calling `get_ontop`), whereas TrivialSimulator defines all
    # 27 getters. One authoritative store beats two partially-overlapping ones.
    getter = "get_" + predicate_name
    if issubclass(base, BinaryAtomicFormula):
        def _evaluate(self, obj1, obj2):
            return bool(getattr(obj1.simulator, getter)((obj1, obj2)))
    else:
        def _evaluate(self, obj):
            return bool(getattr(obj.simulator, getter)((obj,)))
    return type(f"Quiet{base.__name__}", (base,),
                {"_evaluate": _evaluate, "STATE_NAME": predicate_name})


class QuietTrivialBackend(BDDLBackend):
    """bddl's TrivialBackend, silenced and with the STATE_NAME mix-ups corrected."""

    def get_predicate_class(self, predicate_name):
        return _quiet(TrivialBackend().get_predicate_class(predicate_name),
                      predicate_name)


@dataclass
class ToolResult:
    """Structurally identical to ``semantic_tools.ToolResult``.

    Duplicated rather than imported because that module imports OmniGibson at module
    scope, and keeping this file importable without a simulator is the whole point.
    ``rft_sample.result_payload`` duck-types both.
    """

    ok: bool
    tool: str
    args: dict = field(default_factory=dict)
    reason: str = ""
    observation: Optional[dict] = None

    def __repr__(self) -> str:
        return f"<{'OK ' if self.ok else 'REJ'} {self.tool}({self.args}) {self.reason}>"


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class SymbolicWorld:
    """BDDL ground-literal state for one activity, plus the real goal evaluator."""

    def __init__(self, activity: str, definition: int = 0):
        self.activity = activity
        self.definition = definition
        self.definition_source = "installed"
        predefined_problem = None
        definition_root = os.environ.get("BEHAVIOR_BDDL_DEFINITION_ROOT")
        if definition_root:
            problem_path = os.path.join(
                definition_root, activity, f"problem{definition}.bddl"
            )
            if not os.path.isfile(problem_path):
                raise FileNotFoundError(
                    f"pinned BDDL definition missing: {problem_path}"
                )
            predefined_problem = open(problem_path).read()
            # Resolve v3.9 scene-object selectors against the same versioned
            # template population the layout resolver will use. Without this, the
            # old evaluator treats ``cabinet.n.01_*`` as a literal object and adds a
            # guaranteed missing-pose target to the detect denominator.
            from rlinf.envs.behavior.instance_compatibility import (
                expand_problem_wildcards,
                instance_scope,
            )
            from rlinf.envs.behavior.instance_sources import selected_layout_sources

            template_probe = None
            for source in selected_layout_sources():
                candidates = sorted(
                    glob.glob(
                        os.path.join(
                            source.root,
                            "*",
                            "json",
                            f"*_task_{activity}_instances",
                            "*tro_state.json",
                        )
                    )
                )
                if candidates:
                    template_probe = candidates[0]
                    break
            if template_probe:
                predefined_problem = expand_problem_wildcards(
                    predefined_problem, instance_scope(template_probe)
                )
            # The installed v3.7 evaluator calls its equivalent domain
            # ``omnigibson``; v3.9.1 renamed it to ``behavior-1k`` while retaining
            # the predicates this symbolic backend uses. Feed the pinned problem to
            # the old, tested evaluator after changing only the declared domain.
            predefined_problem, replacements = re.subn(
                r"\(:domain\s+behavior-1k\)",
                "(:domain omnigibson)",
                predefined_problem,
                count=1,
            )
            if replacements != 1:
                raise ValueError(
                    f"{problem_path}: expected exactly one behavior-1k domain line"
                )
            self.definition_source = os.path.abspath(problem_path)
        self.conds = Conditions(
            activity,
            definition,
            "omnigibson",
            predefined_problem=predefined_problem,
        )
        self.scope_names = list(get_object_scope(self.conds))
        self.to_display, self.from_display = display_names(self.scope_names)
        # BEHAVIOR always names the robot agent.n.01_1; assert rather than guess,
        # because everything about proximity hangs off finding it.
        agents = [n for n in self.scope_names if synset_of(n) == "agent.n.01"]
        assert len(agents) == 1, f"{activity}: expected one agent, got {agents}"
        self.agent = agents[0]
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.sim = TrivialSimulator()
        self.scope = {n: TrivialGenericObject(n, self.sim) for n in self.scope_names}

        # `TrivialSimulator.set_state` skips `inroom` literals (it has no room
        # concept), so room membership is tracked here -- partial observability and
        # `observe` both need it.
        self.rooms: dict[str, str] = {}
        literals = []
        for lit in self.conds.parsed_initial_conditions:
            if lit and lit[0] == "inroom":
                self.rooms[lit[1]] = lit[2]
            else:
                literals.append(lit)
        self.sim.set_state(literals)

        # `real` starts empty in TrivialSimulator, so without this every object would
        # read as unreal and any `real(...)` goal atom would be trivially false. An
        # object in the scene IS real; only BDDL `future` entries (the products of a
        # not-yet-performed slice/dice/recipe) are not. This mirrors OmniGibson's
        # RealPredicate, which asks whether the scope entry is bound to an object.
        for name in self.scope_names:
            if not self.sim.get_future((self.scope[name],)):
                self.sim.set_real((name,), True)

        self.goal_conditions = get_goal_conditions(
            self.conds, QuietTrivialBackend(), self.scope,
            generate_ground_options=False)
        self._ground_atoms: Optional[list] = None

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def is_real(self, name: str) -> bool:
        return self.sim.get_real((self.scope[name],))

    def support_of(self, name: str):
        """The ``(predicate, target)`` @name rests on/in, or ``None``."""
        for pred in ("inside", "ontop"):
            for a, b in getattr(self.sim, pred):
                if a == name:
                    return pred, b
        return None

    def room_of(self, name: str) -> Optional[str]:
        """Room of @name, following the support chain (a can on a table in the
        kitchen is in the kitchen). Bounded: BDDL support chains are shallow, and a
        malformed cycle must not hang a rollout."""
        seen = set()
        cur = name
        for _ in range(8):
            if cur in self.rooms:
                return self.rooms[cur]
            if cur in seen:
                return None
            seen.add(cur)
            sup = self.support_of(cur)
            if sup is None:
                return None
            cur = sup[1]
        return None

    def clear_position(self, name: str) -> None:
        """Drop every positional relation in which @name is the subject."""
        for pred in _POSITIONAL:
            s = getattr(self.sim, pred)
            for pair in [p for p in s if p[0] == name]:
                self.sim.predicate_to_setters[pred](pair, False)

    def covering_systems(self, name: str) -> list:
        return sorted(b for a, b in self.sim.covered if a == name)

    def enclosing_closed(self, name: str) -> list:
        """Closed openable containers that transitively shut @name away.

        Only ``inside`` encloses -- a pot *on top of* a shut cupboard is in plain
        sight. Bounded like ``room_of`` for the same reason: a malformed support
        cycle must not hang a rollout.
        """
        out, cur, seen = [], name, set()
        for _ in range(8):
            sup = self.support_of(cur)
            if sup is None or cur in seen:
                break
            seen.add(cur)
            pred, parent = sup
            if (pred == "inside" and "openable" in properties_of(parent)
                    and not self.sim.get_open((self.scope[parent],))):
                out.append(parent)
            cur = parent
        return out

    def hosts_of(self, substance: str) -> list:
        """Objects a substance system fills or covers -- the only location a
        substance has. An empty list means the system is not localized anywhere,
        which is the normal state for a system the task has yet to use."""
        return sorted({a for a, b in self.sim.filled if b == substance}
                      | {a for a, b in self.sim.covered if b == substance})

    def holds(self, predicate: str, args: list) -> bool:
        """Evaluate ONE ground atom against the current literal set.

        Used by the expert planner to skip atoms that are already true, so a
        `not ontop(x, floor)` emitted after an `inside(x, box)` that already cleared
        it does not pointlessly pick the object back up.
        """
        getter = getattr(self.sim, f"get_{predicate}", None)
        if getter is None or any(a not in self.scope for a in args):
            return False
        return bool(getter(tuple(self.scope[a] for a in args)))

    def goal_status(self) -> dict:
        """Real BDDL goal evaluation -- pure logic over the literal set.

        Same shape as ``SemanticACI.goal_status`` so the reward module and the env
        wrapper are backend-agnostic.
        """
        if not self.goal_conditions:
            return {"satisfied": [], "unsatisfied": []}
        _, status = evaluate_goal_conditions(self.goal_conditions)
        return {"satisfied": list(status.get("satisfied", [])),
                "unsatisfied": list(status.get("unsatisfied", []))}

    def is_success(self) -> bool:
        gs = self.goal_status()
        return len(gs["unsatisfied"]) == 0 and len(gs["satisfied"]) > 0

    def ground_goal_atoms(self) -> list:
        """Ground atoms of the FIRST ground goal option, as nested lists.

        ``["inside", "can__of__soda.n.01_1", "ashcan.n.01_1"]`` or, negated,
        ``["not", ["open", "electric_refrigerator.n.01_1"]]``. Same source and same
        ``[0]`` choice as the OmniGibson path's ``task.ground_goal_state_options[0]``,
        so the prompt's goal block is identical across backends.

        A quantified goal can have several ground options (9 for
        ``putting_away_Halloween_decorations``) and only ONE need hold. Taking [0]
        is therefore a prompt-rendering choice, not the scoring rule -- scoring stays
        with ``goal_status`` over the unground conditions, which accepts any option.

        Compiled lazily: ``generate_ground_options=True`` enumerates the quantifier
        expansions, and nothing on the rollout path needs it.
        """
        if self._ground_atoms is None:
            grounded = get_goal_conditions(self.conds, QuietTrivialBackend(),
                                           self.scope, generate_ground_options=True)
            options = get_ground_goal_state_options(
                self.conds, QuietTrivialBackend(), self.scope, grounded)
            self._ground_atoms = [list(getattr(a, "body", a)) for a in options[0]]
        return self._ground_atoms

    def goal_predicates(self) -> set:
        """Predicate names appearing anywhere in the compiled goal."""
        from bddl.logic_base import AtomicFormula

        out: set = set()

        def walk(node):
            if isinstance(node, AtomicFormula):
                out.add(node.STATE_NAME)
                return
            for child in getattr(node, "children", []) or []:
                walk(child)

        for head in self.goal_conditions or []:
            walk(head)
        return out


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #
class SymbolicACI:
    """The 17 semantic tools as precondition/effect operators over a SymbolicWorld.

    Method names, argument names and ``ToolResult`` shape match ``SemanticACI``
    exactly, so ``rft_sample.result_payload``, the SFT tool schemas and the agent
    loop are unchanged and the SFT cold start transfers.
    """

    # The observation modes form a 2x2 over the two things `detect` changed at once.
    # `detect` measured -0.677 against `fov` (0816), but it moved BOTH axes, so the
    # two middle cells exist to price them separately:
    #
    #                    addressed by NAME      addressed by HANDLE/BOX
    #   scope only       fov                    detect_scope
    #   + furniture      fov_distract           detect
    #
    # Reading down a column isolates DISTRACTORS; reading across a row isolates
    # REFERENCE RESOLUTION. Occlusion belongs to the detect family only, so it is
    # constant within each row and within each column -- never a difference term.
    CAMERA_MODES = ("fov", "fov_distract", "detect", "detect_scope")
    DETECT_MODES = ("detect", "detect_scope")          # handle/box selection
    DISTRACTOR_MODES = ("fov_distract", "detect")      # furniture is REPORTED
    OBS_MODES = ("full", "partial", "object") + CAMERA_MODES

    def __init__(self, world: SymbolicWorld, obs_mode: str = "full",
                 layout_prefer: str = "sampled", reason_mode: str = "terse"):
        assert obs_mode in self.OBS_MODES, obs_mode
        assert reason_mode in ("verbose", "terse", "silent"), reason_mode
        self.reason_mode = reason_mode
        self.world = world
        self.obs_mode = obs_mode
        self._held: Optional[str] = None          # scope name
        self._resolve_error: Optional[str] = None  # why the last selection failed
        # Start where the agent's own initial condition puts it, so the first
        # `observe` is not an empty room in partial mode.
        sup = world.support_of(world.agent)
        self._at: Optional[str] = sup[1] if sup else None

        # `fov` is the only mode that needs metric coordinates, so the layout and the
        # camera are built only for it. Every other mode stays byte-identical, which
        # is what keeps the 0801 baselines reproducible.
        self.layout = None
        self.view = None
        self._dets: list = []          # detect modes: last observation
        self._det_audit: dict[str, set[str]] = {}  # cumulative detector-stage evidence
        self._view_epoch = 0           # bumped by any camera mutation
        self._det_epoch = -1           # epoch the handles in `_dets` belong to
        self._furn: dict = {}          # fov_distract: display name -> SceneObject
        if obs_mode in self.CAMERA_MODES:
            from rlinf.envs.behavior.layout import build_layout
            from rlinf.envs.behavior.viewpoint import Viewpoint
            self.layout = build_layout(
                world.activity,
                world,
                prefer=layout_prefer,
                # Detect needs the paired template to recover the concrete asset
                # model for each scope name. Local tro_state-only samples cannot
                # satisfy that contract and must not fall back to category medians.
                require_detect_models=obs_mode in self.DETECT_MODES,
            )
            if obs_mode != "fov" and not self.layout.is_real:
                # NO INVENTED POSES ONCE THE SCENE IS INVOLVED. Every pose must come
                # from BEHAVIOR's sampler, which is the only source that puts task
                # objects in the SCENE's frame and satisfies the BDDL initial
                # conditions geometrically. A generated layout satisfies neither:
                # `inside(can, ashcan)` can hold symbolically while the can sits three
                # metres away, and mixing invented task poses with real furniture makes
                # every occlusion and every bearing an accident. Plain `fov` tolerates
                # a generated layout because it reports no furniture and claims no
                # scene; the other three all read the scene json and cannot.
                raise ValueError(
                    f"obs_mode={obs_mode} requires a sampled instance for "
                    f"{world.activity!r}; none found. Generate one with "
                    f"rlinf/envs/behavior/instance_generator.py.")
            self.view = Viewpoint(x=self.layout.robot_xy[0], y=self.layout.robot_xy[1])
            if obs_mode == "fov_distract":
                self._furn = self._name_furniture()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _name_furniture(self) -> dict:
        """Scene furniture -> the display names `fov_distract` reports it under.

        The scene json calls a cabinet `bottom_cabinet_pkdnbu_0` -- a name carrying an
        asset model id, which a scope object's `ashcan_1` never does. Reporting that
        raw would hand the policy a FORMAT that separates distractor from task object
        without reading either, and the arm would measure nothing. So furniture is
        renamed into exactly the scope's `<category>_<n>` shape, with the counter
        skipping any name the scope already claims -- an `ashcan_1` in the goal and an
        `ashcan_1` on the floor would be a genuine ambiguity we did not intend to test.

        This renames; it invents nothing. Category, room, pose and open state below are
        all read from the shipped scene json.
        """
        from rlinf.envs.behavior.detect import scene_furniture

        out: dict = {}
        counter: dict = {}
        for so in scene_furniture(self.layout.scene or ""):
            n = counter.get(so.category, 0)
            while True:
                n += 1
                disp = f"{so.category}_{n}"
                if disp not in self.world.from_display and disp not in out:
                    break
            counter[so.category] = n
            out[disp] = so
        return out

    def _resolve(self, name) -> Optional[str]:
        """Selection -> scope name, or None.

        In the `detect` family the argument is a view-local handle (`"d3"`) or a box
        (`"118,96,171,148"`), never an object name -- names are the leak those modes
        exist to remove. Routing it through here rather than through each tool is
        what lets all 17 tool signatures stay unchanged, so expert plans and the SFT
        schema are untouched.

        A failed spatial resolution leaves `self._resolve_error` set with the reason,
        because "ambiguous" and "no such object" are different failures and the policy
        needs to be able to tell them apart.
        """
        self._resolve_error = None
        if not isinstance(name, str):
            return None
        if self.obs_mode in self.DETECT_MODES:
            from rlinf.envs.behavior.detect import resolve as _resolve_det
            if self._det_epoch != self._view_epoch:
                self._resolve_error = ("your last observation is stale; "
                                       "observe again before selecting")
                return None
            det, err = _resolve_det(name, self._dets)
            if det is None:
                self._resolve_error = err
                return None
            # Scene furniture is selectable and real, but has no BDDL identity, so no
            # tool can act on it. Saying so is honest; silently treating it as
            # "no such object" would hide that the policy pointed at a real thing.
            # Unreachable under `detect_scope`, which reports no furniture -- kept
            # anyway so the two modes differ in what is REPORTED and nothing else.
            if not det.key.startswith("scope:"):
                self._resolve_error = (f"that is a {det.category}; it is scene furniture "
                                       f"and no tool applies to it")
                return None
            return det.key[len("scope:"):]
        if name in self.world.from_display:
            return self.world.from_display[name]
        if name in self.world.scope:
            return name
        # Furniture is namable under `fov_distract` and is refused in the same words
        # `detect` uses, so a distractor costs the policy the same acknowledged dead
        # end in both. Checked AFTER the scope, so a name the scope claims is never
        # shadowed.
        if name in self._furn:
            self._resolve_error = (f"that is a {self._furn[name].category}; it is scene "
                                   f"furniture and no tool applies to it")
            return None
        return None

    def _disp(self, scope_name: Optional[str]) -> Optional[str]:
        return self.world.to_display.get(scope_name) if scope_name else None

    def _ground(self) -> Optional[str]:
        """What the robot is standing on: the support of whatever it navigated to,
        else that object itself. Where a released object lands."""
        if self._at is None:
            return None
        sup = self.world.support_of(self._at)
        return sup[1] if sup else self._at

    def _is_near(self, name: str) -> bool:
        """Relational proximity -- see the module docstring, gap 2."""
        if name == self._held or name == self._at:
            return True
        if self._at is None:
            return False
        sup_obj = self.world.support_of(name)
        if sup_obj and sup_obj[1] == self._at:
            return True
        sup_at = self.world.support_of(self._at)
        return bool(sup_at and sup_at[1] == name)

    # `SemanticACI` exposes these directly, and callers such as
    # `rft_sample.rollout_once` reach for them on the ACI rather than the world.
    # Delegating keeps the two ACIs substitutable.
    def goal_status(self) -> dict:
        return self.world.goal_status()

    def is_success(self) -> bool:
        return self.world.is_success()

    def _result(self, ok, tool, args, reason) -> ToolResult:
        # In `detect` mode an action ack carries NO observation. Two reasons, and the
        # second is a correctness bug this fixes: a 26-detection payload on every ack
        # is the context cost we are trying to make the policy *choose* to pay, and
        # re-running `observe()` here silently re-issued the view-local handles, so
        # `turn_left` refreshed the very handles it was supposed to invalidate.
        obs = None if self.obs_mode in self.DETECT_MODES else self.observe()
        return ToolResult(ok=ok, tool=tool, args=args,
                          reason=reason if ok else self._refusal(reason),
                          observation=obs)

    def _refusal(self, reason: str) -> str:
        """Apply `reason_mode` to a refusal. Every refusal funnels through here.

        E1 (0815) found that stripping the procedural system prompt costs nothing,
        because the policy simply reads the same knowledge out of our refusals
        instead: `minimal` walked into `not near sink_1; go_to(sink_1) first` and
        recovered from it. So the refusal text is a second skill channel, and its
        halves supply different things:

            precondition failed: not near sink_1;  go_to(sink_1) first
            |____ state: which precondition ____|  |__ plan: which tool __|

        `verbose` gives both, `terse` gives the state only, `silent` gives neither
        (ALFWorld's "Nothing happens", which its own paper blames for the repeat
        loops it then patched with beam search). Splitting on the repair clause
        rather than removing feedback wholesale is what lets us price the plan half
        separately from the diagnosis half -- and keep the half that is a
        contribution while dropping the half that is a leak.
        """
        if self.reason_mode == "verbose":
            return reason
        if self.reason_mode == "silent":
            return ""
        return reason.split("; ")[0]

    def _future_reason(self, name: str, scope_name: str) -> str:
        """Why @name does not exist yet, AND what would create it.

        Measured on the 164-episode run: 63% of episodes hit at least one refusal but
        only 10% ever repeated an identical one -- except for this message, which
        accounted for **all five** of the worst repeat-loops (up to 5 identical calls
        in one episode). The refusals that name a repair (`not near X; go_to(X)
        first`) did not loop. Stating the repair is the whole mechanism, so the one
        refusal that omitted it is the one the policy got stuck on.
        """
        made = producer_of(self.world, scope_name)
        if made is None:
            return (f"{name} does not exist yet (it is a future object) and no tool "
                    f"in this environment creates it")
        whole, how = made
        return (f"{name} does not exist yet (it is a future object); "
                f"{how}({self._disp(whole)}) creates it")

    def _require(self, tool, args, name, prop=None, prop_msg=""):
        """Resolve @name and check existence / realness / proximity / property.

        Returns ``(scope_name, None)`` on success or ``(None, ToolResult)`` on the
        first failed precondition. Every tool funnels through this so refusal
        messages stay uniform and no tool silently skips a check.
        """
        scope_name = self._resolve(name)
        if scope_name is None:
            return None, self._result(
                False, tool, args,
                self._resolve_error or f"no such object {name!r}")
        if not self.world.is_real(scope_name):
            return None, self._result(False, tool, args, self._future_reason(name,
                                                                            scope_name))
        if not self._is_near(scope_name):
            return None, self._result(
                False, tool, args,
                f"precondition failed: not near {name}; go_to({name}) first")
        # You cannot reach through a shut door. Without this the `object` obs mode is
        # decorative: an `atoms` goal names the object, so a policy could grasp it out
        # of a closed fridge it was never shown. This is a DYNAMICS change and applies
        # in every obs mode -- observability and reachability must not disagree.
        shut = self.world.enclosing_closed(scope_name)
        if shut:
            return None, self._result(
                False, tool, args,
                f"precondition failed: {name} is shut inside "
                f"{self._disp(shut[0])}; open({self._disp(shut[0])}) first")
        if prop is not None and prop not in properties_of(scope_name):
            return None, self._result(False, tool, args,
                                      f"precondition failed: {name} is not {prop_msg}")
        return scope_name, None

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def _obj_report(self, name: str) -> dict:
        props = properties_of(name)
        sim = self.world.sim
        obj = self.world.scope[name]

        states: dict = {}
        if "openable" in props:
            states["Open"] = bool(sim.get_open((obj,)))
        if "toggleable" in props:
            states["ToggledOn"] = bool(sim.get_toggled_on((obj,)))
        if "cookable" in props:
            states["Cooked"] = bool(sim.get_cooked((obj,)))
        covered = self.world.covering_systems(name)
        if covered:
            states["Covered"] = [self._disp(c) or c for c in covered]

        report = {
            "name": self._disp(name),
            "category": _word(synset_of(name)),
            "is_near": self._is_near(name),
            "states": states,
        }
        # No coordinates: there is no geometry here, and inventing an (x, y) would be
        # exactly the kind of plausible-looking fake the honesty rules forbid. The
        # support relation is the real spatial information this layer has.
        sup = self.world.support_of(name)
        if sup:
            report["on_top_of" if sup[0] == "ontop" else "inside_of"] = self._disp(sup[1])
        room = self.world.room_of(name)
        if room:
            report["room"] = room
        return report

    def _visible(self, name: str, here: Optional[str], seen=()) -> bool:
        """Is @name reported by `observe()` under the current mode?

        `full`    -- everything real.
        `partial` -- room-scoped, kept BYTE-IDENTICAL so the 0801 numbers stay
                     reproducible. Measured, it hides very little: 87.7% of the 740
                     solvable activities are single-room, and 81.6% of what it does
                     hide is hidden only because the object has no room at all.
        `object`  -- room-scoped AND shut containers hide their contents, which is
                     what forces a search (`open` the cupboard to find the pan)
                     rather than reading a full manifest at reset.

        `object` is deliberately NOT a strict superset of `partial`'s hiding. What
        `partial` mostly hides is substance systems, and it hides them because
        `room_of` returns None for a thing with no support -- an artifact of the
        annotation, not a model of occlusion. Here a substance is located by what it
        fills or covers, and an unlocalized system stays visible because in
        BEHAVIOR a system is scene-global, not an object sitting somewhere.
        """
        if name == self._held or name == self._at:
            return True
        if self.obs_mode == "full":
            return True
        if self.obs_mode in ("fov", "fov_distract"):
            # Shut containers hide their contents here too: a camera cannot see
            # through a closed fridge door any more than `object` mode can.
            if self.world.enclosing_closed(name):
                return False
            pos = self.layout.pos(name)
            if pos is None:
                # No coordinates -- substances with no host. They are scene-global in
                # BEHAVIOR rather than objects sitting somewhere, so a viewpoint gate
                # has nothing to test and hiding them would be an artifact, not
                # occlusion. Same reasoning as the `object` branch below.
                return True
            return self.view.in_view(pos)
        if self.obs_mode == "object":
            if self.world.enclosing_closed(name):
                return False
            if "substance" in properties_of(name):
                # Bounded like `room_of` / `enclosing_closed`: hosts are objects, not
                # systems, so this cannot actually cycle -- but a hang is the one
                # failure mode this file refuses to leave reachable.
                hosts = [h for h in self.world.hosts_of(name) if h not in seen]
                return (not self.world.hosts_of(name)) or any(
                    self._visible(h, here, (*seen, name)) for h in hosts)
        return self.world.room_of(name) == here

    def observe(self) -> dict:
        """Symbolic observation of task-relevant objects. Never leaks the goal.

        Observability is gated by `_visible`; the three modes and why `object`
        exists are documented there.
        """
        if self.obs_mode in self.DETECT_MODES:
            return self._observe_detect()
        here = self.world.room_of(self._at) if self._at else None
        names = [n for n in self.world.scope_names
                 if n != self.world.agent and self.world.is_real(n)
                 and self._visible(n, here)]
        objs = [self._obj_report(name) for name in names]
        out = {
            "obs_mode": self.obs_mode,
            "held": self._disp(self._held),
            "at": self._disp(self._at),
            "room": here,
            "objects": objs,
        }
        if self.obs_mode in ("fov", "fov_distract"):
            # Remember what has been seen: `go_to` refuses an unseen target, so this
            # set is the record of what discovery has bought. Held and current-anchor
            # objects come through `_visible` too, so they land here as well.
            self.view.seen.update(names)
            out["view"] = self.view.state()
            metric = self.layout.is_real
            for rep, name in zip(objs, names):
                pos = self.layout.pos(name)
                if pos is not None:
                    rep["where"] = self.view.describe(pos, metric)
            # Say which it is. A policy reading "2.3 m" off a generated layout would
            # be reading an invention; under `generated` no metre value is emitted at
            # all, and this field says why.
            out["geometry"] = "measured" if metric else "schematic"
            if self.obs_mode == "fov_distract":
                out["objects"] = self._merge_furniture(objs, names)
        return out

    def _merge_furniture(self, objs: list, names: list) -> list:
        """`fov_distract`: the scene's own furniture, reported alongside the scope.

        This is `detect`'s distractor axis WITHOUT its selection axis -- furniture
        arrives mixed in, and everything is still addressed by name. Together with
        `detect_scope` it splits the -0.677 that `detect` measured as one number.

        Sorted by distance, both halves together. Appending furniture as a trailing
        block would let the policy find the task objects by POSITION in the list and
        never read a category, which is the whole thing this arm is trying to make it
        do. Plain `fov` keeps its original order untouched, because its 0.871 is a
        reported baseline; that ordering difference is a real, if small, second
        difference between the two arms and is recorded as one.
        """
        rows = list(objs)
        for name, so in self._furn.items():
            if not self.view.in_view(so.pos):
                continue
            # Key ORDER matters, not just key set: these dicts are serialised to JSON
            # with insertion order preserved, so a furniture row whose `where` came
            # before its `room` would be separable from a scope row by field order
            # alone -- a tell that needs no reading of any value. Same reason
            # `in_rooms` loses its index below: the scope side reports `living_room`
            # from the BDDL annotation, and `living_room_0` would be a giveaway of a
            # different kind. Both are presentation, and neither changes a fact.
            rep = {"name": name, "category": so.category, "is_near": False,
                   "states": ({"Open": so.is_open} if so.openable else {})}
            if so.room:
                rep["room"] = so.room.rsplit("_", 1)[0] if so.room[-1].isdigit() \
                    else so.room
            rep["where"] = self.view.describe(so.pos, self.layout.is_real)
            rows.append(rep)
        pos_of = {n: self.layout.pos(n) for n in names}

        def _key(rep):
            p = (self._furn[rep["name"]].pos if rep["name"] in self._furn
                 else pos_of.get(self.world.from_display.get(rep["name"])))
            # No coordinates (an unlocalized substance) sorts last rather than
            # crashing or being given an invented distance.
            return (1, 0.0) if p is None else (0, self.view.range_to(p))

        rows.sort(key=_key)
        return rows

    def _observe_detect(self) -> dict:
        """Detections in view -- scene furniture and task objects, undifferentiated.

        This is the method that removes the object-scope oracle. `full`/`partial`/
        `object`/`fov` all report the BDDL scope: median 9 objects out of the 260 in a
        real scene, and precisely the 9 the task needs. Here the task objects arrive
        mixed with the furniture and working out which is which is the policy's job.

        No name, no instance index, no `is_near`, no `Cooked` -- `0815 - interface
        audit` records why none of those could come from a camera.

        `detect_scope` reports the scope only, and is the control that prices the
        selection mechanism on its own. Furniture still enters the projection there
        and still OCCLUDES; it is dropped after `detect()` has run, so the two modes
        differ in what is reported and in nothing else. Removing furniture from the
        entry list instead would have quietly removed every occluder too, and the
        comparison would have been against a mode that is easier for a second reason.
        """
        from rlinf.envs.behavior.detect import (
            detect,
            extent_for_category,
            extent_for_model,
            offset_for_model,
            scene_furniture,
            scope_assets,
            scope_models,
            scope_scene_names,
            world_bbox,
        )

        assets = scope_assets(self.layout.instance_path or "")
        models = scope_models(self.layout.instance_path or "")
        bound_scene_names = scope_scene_names(self.layout.instance_path or "")

        entries = []
        for name in self.world.scope_names:
            if name == self.world.agent or not self.world.is_real(name):
                continue
            audit_key = f"scope:{name}"
            self._det_audit.setdefault(audit_key, set()).add("target")
            if self.world.enclosing_closed(name):
                self._det_audit[audit_key].add("closed_container")
                continue                  # a shut container hides its contents
            pos = self.layout.pos(name)
            if pos is None:
                event = (
                    "nonvisual_substance"
                    if "substance" in properties_of(name)
                    else "missing_pose"
                )
                self._det_audit[audit_key].add(event)
                continue                  # substances have no location by construction
            cat = _word(synset_of(name))
            # Size comes from the asset, per MODEL where the instance recorded one and
            # per category otherwise -- never from a hand-written table. An object we
            # cannot size is omitted rather than guessed, because a wrong size is
            # silently wrong: it changes what occludes what and what is visible from
            # where.
            asset = assets.get(name)
            scale = asset["scale"] if asset else (1.0, 1.0, 1.0)
            ext = (extent_for_model(models[name]) if name in models else None) \
                or extent_for_category(cat)
            if ext is None:
                self._det_audit[audit_key].add("missing_extent")
                continue
            # The instance file records the BASE LINK pose; the bbox is centred at
            # pose + the ROTATED ig:offsetBaseLink. Its local extents must likewise
            # be rotated into a conservative world AABB.
            off = offset_for_model(models[name]) if name in models else (0.0, 0.0, 0.0)
            centre, ext = world_bbox(
                pos,
                ext,
                local_offset=off,
                scale=scale,
                orientation=self.layout.orientation.get(name, (0.0, 0.0, 0.0, 1.0)),
            )
            entries.append((f"scope:{name}", cat, centre, ext))
        for i, so in enumerate(scene_furniture(self.layout.scene or "")):
            if so.name in bound_scene_names:
                # The same physical object is already represented by its scope
                # entry. Keeping both boxes makes an exact self-occluder and can
                # prevent a navigated-to cabinet from ever being reacquired.
                continue
            entries.append((f"scene:{i}", so.category, so.pos, so.extent))

        # An AABB is a coarse outer bound, not a solid object. A cabinet or table's
        # box must not hide the objects BDDL says are inside/on it; otherwise opening
        # a container changes the symbolic state while its full AABB continues to
        # occlude every content forever. Exempt the complete support ancestry while
        # preserving occlusion from unrelated furniture and sibling objects.
        exempt_pairs = set()
        for name in self.world.scope_names:
            child = name
            seen = set()
            while child not in seen:
                seen.add(child)
                support = self.world.support_of(child)
                if support is None:
                    break
                parent = support[1]
                exempt_pairs.add((f"scope:{name}", f"scope:{parent}"))
                child = parent

        dets = detect(
            self.view,
            entries,
            audit=self._det_audit,
            occlusion_exempt_pairs=frozenset(exempt_pairs),
        )
        if self.obs_mode == "detect_scope":
            kept = [d for d in dets if d.key.startswith("scope:")]
            # Renumber, or the handles carry gaps (d2, d5, d9) that leak how many
            # detections were withheld -- and the count of hidden furniture is a
            # signal `detect` does not give either.
            dets = [dataclasses.replace(d, det=f"d{i + 1}")
                    for i, d in enumerate(kept)]
        self._dets = dets
        self._det_epoch = self._view_epoch
        return {
            "obs_mode": self.obs_mode,
            "view": self.view.state(),
            "held": bool(self._held),
            "geometry": "measured" if self.layout.is_real else "schematic",
            "detections": [
                {"det": d.det, "bbox": list(d.bbox), "category": d.category,
                 "score": d.score}
                for d in self._dets
            ],
        }

    # ------------------------------------------------------------------ #
    # navigation
    # ------------------------------------------------------------------ #
    def go_to(self, name: str) -> ToolResult:
        args = {"name": name}
        scope_name = self._resolve(name)
        if scope_name is None:
            return self._result(False, "go_to", args,
                                self._resolve_error or f"no such object {name!r}")
        if not self.world.is_real(scope_name):
            return self._result(False, "go_to", args,
                                self._future_reason(name, scope_name))
        if self.obs_mode in ("fov", "fov_distract") and scope_name not in self.view.seen:
            # You cannot navigate to what you have not found. This is the whole point
            # of the mode: navigation stays an oracle, DISCOVERY does not. Without it
            # the policy reads a name out of the goal and teleports to it, and the
            # camera is decoration.
            return self._result(
                False, "go_to", args,
                f"you have not seen {name} yet; look around first "
                f"(turn_left / turn_right / move_ahead / look_down, then observe)")
        self._at = scope_name
        # Keep the agent's own literal coherent with `_at`; nothing reads it today,
        # but a state dump that disagrees with the robot's position is a trap.
        ground = self._ground()
        self.world.clear_position(self.world.agent)
        if ground is not None and ground != self.world.agent:
            self.world.sim.set_ontop((self.world.agent, ground), True)
        if self.obs_mode in self.CAMERA_MODES:
            # Arriving puts the camera at the object, facing it. Navigation is still
            # the oracle; only finding the target was work.
            #
            # THIS USED TO BE `fov` ONLY, and that was an omission rather than a
            # decision. Under `detect` the robot navigated symbolically while the
            # camera stayed where it was, so reaching a new part of the house meant
            # walking it in 0.5 m `move_ahead` steps -- 17 of 31 episodes hit the
            # 400-turn budget at a mean of 266 steps, and that number was pricing
            # NAVIGATION, not identification. The two families now share one
            # navigation model, which is what makes a cross-family comparison mean
            # anything. It also supersedes the `detect` = 0.194 of 0816; that arm is
            # re-run rather than quoted.
            pos = self.layout.pos(scope_name)
            if pos is not None:
                # Stop in front of the object, not at its bbox centre. The old
                # exact-centre teleport put the camera inside cabinets/fridges and
                # made the handle impossible to reacquire after navigation -- which
                # in turn made every privileged-tour open() fail.
                dx, dy = self.view.x - pos[0], self.view.y - pos[1]
                norm = math.hypot(dx, dy)
                if norm < 1e-6:
                    dx = -math.cos(self.view.yaw)
                    dy = -math.sin(self.view.yaw)
                    norm = 1.0
                # Stand outside the concrete scaled AABB. A fixed 0.9 m stop put
                # the camera *inside* v3.9 cabinets scaled beyond 2x, so the tour
                # could detect them from afar but never reacquire/open them nearby.
                from rlinf.envs.behavior.detect import extent_for_model, scope_assets

                asset = scope_assets(self.layout.instance_path or "").get(scope_name)
                stop = 0.9
                if asset:
                    native = extent_for_model(asset["model"])
                    if native:
                        scaled_x = native[0] * asset["scale"][0]
                        scaled_y = native[1] * asset["scale"][1]
                        stop = max(stop, math.hypot(scaled_x, scaled_y) + 0.3)
                camera_x = pos[0] + stop * dx / norm
                camera_y = pos[1] + stop * dy / norm
                yaw = math.atan2(pos[1] - camera_y, pos[0] - camera_x)
                self.view.teleport(camera_x, camera_y, yaw)
                if self.obs_mode in self.DETECT_MODES:
                    # The camera moved, so every handle in the last observation
                    # refers to a frame that no longer exists.
                    self._view_epoch += 1
        return self._result(True, "go_to", args, f"now at {name}")

    # ------------------------------------------------------------------ #
    # camera control -- `fov` mode only
    # ------------------------------------------------------------------ #
    def _camera(self, tool: str) -> ToolResult:
        """The five viewpoint primitives.

        They mutate the camera and NOTHING else: no BDDL literal changes, so every
        expert plan stays valid and the 17 semantic tools keep their meaning. The only
        effect is on what the next `observe()` reports -- which is the point.
        """
        if self.obs_mode not in self.CAMERA_MODES:
            return self._result(False, tool, {},
                                f"{tool} needs obs_mode="
                                f"{'|'.join(self.CAMERA_MODES)}")
        getattr(self.view, tool)()
        self._view_epoch += 1          # every det handle just went stale
        st = self.view.state()
        return self._result(True, tool, {},
                            f"facing {st['facing_deg']} deg, "
                            f"pitch {st['pitch_deg']} deg")

    def turn_left(self) -> ToolResult:
        return self._camera("turn_left")

    def turn_right(self) -> ToolResult:
        return self._camera("turn_right")

    def move_ahead(self) -> ToolResult:
        return self._camera("move_ahead")

    # WASD: translate without turning. `turn, walk, turn back` reaches the same place
    # in 4 calls and loses the framing in between, which is exactly the manoeuvre a
    # policy needs when an object is occluded from where it stands.
    def move_back(self) -> ToolResult:
        return self._camera("move_back")

    def strafe_left(self) -> ToolResult:
        return self._camera("strafe_left")

    def strafe_right(self) -> ToolResult:
        return self._camera("strafe_right")

    def look_up(self) -> ToolResult:
        return self._camera("look_up")

    def look_down(self) -> ToolResult:
        return self._camera("look_down")

    # ------------------------------------------------------------------ #
    # manipulation
    # ------------------------------------------------------------------ #
    def grasp(self, name: str) -> ToolResult:
        args = {"name": name}
        if self._held is not None:
            return self._result(False, "grasp", args,
                                f"precondition failed: already holding "
                                f"{self._disp(self._held)}")
        scope_name, rej = self._require("grasp", args, name)
        if rej is not None:
            return rej
        props = properties_of(scope_name)
        # The geometric ACI would happily `set_position_orientation` a fixed-base
        # table; refusing is stricter, not looser, so it cannot manufacture success.
        if "sceneObject" in props:
            return self._result(False, "grasp", args,
                                f"precondition failed: {name} is fixed scene furniture")
        if "substance" in props:
            return self._result(False, "grasp", args,
                                f"precondition failed: {name} is a substance, not a "
                                f"graspable object")
        self.world.clear_position(scope_name)
        self.world.sim.set_grasped((self.world.agent, scope_name), True)
        self._held = scope_name
        return self._result(True, "grasp", args, f"holding {name}")

    def release(self, name: str | None = None) -> ToolResult:
        if self._held is None:
            return self._result(False, "release", {},
                                "precondition failed: not holding anything")
        held = self._held
        ground = self._ground()
        self.world.sim.set_grasped((self.world.agent, held), False)
        if ground is not None and ground != held:
            self.world.sim.set_ontop((held, ground), True)
        self._held = None
        return self._result(True, "release", {"name": self._disp(held)},
                            f"released {self._disp(held)} onto {self._disp(ground)}")

    def _place(self, tool, name, target, pred) -> ToolResult:
        args = {"name": name, "surface" if pred == "ontop" else "container": target}
        movable = self._resolve(name)
        if movable is None or movable != self._held:
            return self._result(False, tool, args,
                                f"precondition failed: not holding {name} "
                                f"(held={self._disp(self._held)})")
        scope_target, rej = self._require(tool, args, target)
        if rej is not None:
            return rej
        if movable == scope_target:
            return self._result(False, tool, args,
                                f"cannot place {name} on/in itself")
        tprops = properties_of(scope_target)
        if "substance" in tprops:
            return self._result(False, tool, args,
                                f"precondition failed: {target} is a substance, "
                                f"not a support")
        # An openable container must be open. The geometric layer got this for free
        # (a closed fridge has no interior volume to sample into); symbolically it
        # has to be stated, and it is what forces open-then-place plans.
        if pred == "inside" and "openable" in tprops and not self.world.sim.get_open(
                (self.world.scope[scope_target],)):
            return self._result(False, tool, args,
                                f"precondition failed: {target} is closed; "
                                f"open({target}) first")
        self.world.clear_position(movable)
        self.world.sim.set_grasped((self.world.agent, movable), False)
        self.world.sim.predicate_to_setters[pred]((movable, scope_target), True)
        if pred == "ontop":
            self.world.sim.set_nextto((movable, scope_target), True)
        self._held = None
        return self._result(True, tool, args, f"{pred}({name}, {target})=True")

    def place_on(self, name: str, surface: str) -> ToolResult:
        return self._place("place_on", name, surface, "ontop")

    def place_inside(self, name: str, container: str) -> ToolResult:
        return self._place("place_inside", name, container, "inside")

    # ------------------------------------------------------------------ #
    # flag states
    # ------------------------------------------------------------------ #
    def _set_open(self, tool, name, value) -> ToolResult:
        args = {"name": name}
        scope_name, rej = self._require(tool, args, name, "openable", "openable")
        if rej is not None:
            return rej
        # TrivialSimulator keeps `open` and `closed` as INDEPENDENT sets, so setting
        # only one leaves the other stale and a `not closed` goal atom reads wrong.
        self.world.sim.set_open((scope_name,), value)
        self.world.sim.set_closed((scope_name,), not value)
        return self._result(True, tool, args, f"Open={value}")

    def open(self, name: str) -> ToolResult:
        return self._set_open("open", name, True)

    def close(self, name: str) -> ToolResult:
        return self._set_open("close", name, False)

    def _set_toggled(self, tool, name, value) -> ToolResult:
        args = {"name": name}
        scope_name, rej = self._require(tool, args, name, "toggleable", "toggleable")
        if rej is not None:
            return rej
        self.world.sim.set_toggled_on((scope_name,), value)
        return self._result(True, tool, args, f"ToggledOn={value}")

    def toggle_on(self, name: str) -> ToolResult:
        return self._set_toggled("toggle_on", name, True)

    def toggle_off(self, name: str) -> ToolResult:
        return self._set_toggled("toggle_off", name, False)

    def cook(self, name: str) -> ToolResult:
        args = {"name": name}
        scope_name, rej = self._require("cook", args, name, "cookable", "cookable")
        if rej is not None:
            return rej
        self.world.sim.set_cooked((scope_name,), True)
        return self._result(True, "cook", args, "Cooked=True")

    # ------------------------------------------------------------------ #
    # substances
    # ------------------------------------------------------------------ #
    def _resolve_system(self, tool, args, system_name):
        """Resolve a substance argument, accepting a CATEGORY as well as an instance.

        The geometric ACI's `system_name` is an OmniGibson *system* name --
        `scene.get_system("dust")` -- with no instance index, so the SFT cold start
        was trained to pass `"dust"`. Requiring `dust_1` here was a divergence I
        introduced, and it was the single biggest source of failure in the first S2
        run: 70+ rejections across `dust`, `stain`, `water`, `disinfectant`,
        `mud`, `crumb`, ... on a policy that was doing the right thing.

        A bare category is only accepted when it names exactly ONE substance in
        scope. Several, and it is ambiguous -- refuse and list them rather than pick.
        """
        scope_name = self._resolve(system_name)
        if scope_name is None:
            matches = [n for n in self.world.scope_names
                       if lemma_of(n) == system_name
                       and "substance" in properties_of(n)]
            if len(matches) > 1:
                return None, self._result(
                    False, tool, args,
                    f"ambiguous system {system_name!r}: "
                    f"{[self._disp(m) for m in matches]}; name one exactly")
            if not matches:
                return None, self._result(False, tool, args,
                                          f"no such system {system_name!r}")
            scope_name = matches[0]
        if "substance" not in properties_of(scope_name):
            return None, self._result(
                False, tool, args, f"{system_name} is not a substance system")
        return scope_name, None

    def _set_covered(self, tool, name, system_name, value) -> ToolResult:
        args = {"name": name, "system_name": system_name}
        scope_name, rej = self._require(tool, args, name)
        if rej is not None:
            return rej
        system, rej = self._resolve_system(tool, args, system_name)
        if rej is not None:
            return rej
        self.world.sim.set_covered((scope_name, system), value)
        if value:
            # Substances declared `future` become real once they are applied; this is
            # what flips `real(...)` goals on substance-producing activities.
            self.world.sim.set_real((system,), True)
            self.world.sim.set_future((system,), False)
        return self._result(True, tool, args,
                            f"Covered({name}, {system_name})={value}")

    def spray(self, name: str, system_name: str) -> ToolResult:
        return self._set_covered("spray", name, system_name, True)

    def uncover(self, name: str, system_name: str) -> ToolResult:
        return self._set_covered("uncover", name, system_name, False)

    def fill(self, name: str, system_name: str) -> ToolResult:
        args = {"name": name, "system_name": system_name}
        scope_name, rej = self._require("fill", args, name, "fillable", "fillable")
        if rej is not None:
            return rej
        system, rej = self._resolve_system("fill", args, system_name)
        if rej is not None:
            return rej
        # set_filled entails contains (TrivialSimulator.set_state's rule, applied
        # here explicitly because we bypass set_state).
        self.world.sim.set_filled((scope_name, system), True)
        self.world.sim.set_contains((scope_name, system), True)
        self.world.sim.set_real((system,), True)
        self.world.sim.set_future((system,), False)
        return self._result(True, "fill", args, f"Filled({name}, {system_name})=True")

    # ------------------------------------------------------------------ #
    # product-creating transforms
    # ------------------------------------------------------------------ #
    def _materialise(self, prefix: str, base_synset: str, support, limit: int) -> list:
        """Turn up to @limit ``<prefix><base_synset>`` future entries real, seated on
        @support. This is the symbolic stand-in for OmniGibson's SlicingRule /
        DicingRule spawning objects that the BDDL scope then rebinds.

        Matched on the LEMMA, not the full synset: BEHAVIOR does not preserve the
        WordNet sense number across a transform, so ``bell_pepper.n.02`` dices into
        ``diced__bell_pepper.n.01``. Requiring an exact synset match silently
        produced nothing on every such activity.
        """
        target_lemma = f"{prefix}{_word(base_synset)}"
        made = []
        for name in self.world.scope_names:
            if len(made) >= limit:
                break
            if _word(synset_of(name)) != target_lemma or self.world.is_real(name):
                continue
            self.world.sim.set_real((name,), True)
            self.world.sim.set_future((name,), False)
            if support is not None:
                self.world.sim.set_ontop((name, support), True)
            made.append(self._disp(name))
        return made

    def _consume(self, scope_name: str):
        """Remove a transformed object from the world, returning its former support."""
        sup = self.world.support_of(scope_name)
        support = sup[1] if sup else self._ground()
        self.world.clear_position(scope_name)
        if self._held == scope_name:
            self.world.sim.set_grasped((self.world.agent, scope_name), False)
            self._held = None
        self.world.sim.set_real((scope_name,), False)
        return support

    def slice(self, name: str) -> ToolResult:
        args = {"name": name}
        scope_name, rej = self._require("slice", args, name, "sliceable", "sliceable")
        if rej is not None:
            return rej
        base = synset_of(scope_name)
        support = self._consume(scope_name)
        # BEHAVIOR slices one object into exactly two annotated halves.
        made = self._materialise("half__", base, support, limit=2)
        return self._result(bool(made), "slice", args,
                            f"sliced {name} into {made}" if made else
                            f"{name} has no half__ products in this activity's scope")

    def dice(self, name: str) -> ToolResult:
        args = {"name": name}
        scope_name = self._resolve(name)
        if scope_name is None:
            return self._result(False, "dice", args, f"no such object {name!r}")
        props = properties_of(scope_name)
        prop = "diceable" if "diceable" in props else "sliceable"
        scope_name, rej = self._require("dice", args, name, prop,
                                        "diceable or sliceable")
        if rej is not None:
            return rej
        # `diced__X` is named after the WHOLE object even when a half is diced
        # (BEHAVIOR models chopping as slice-then-dice, and the geometric ACI runs
        # both stages), so strip the half__ prefix before looking up the product.
        base = synset_of(scope_name)
        whole = base[len("half__"):] if base.startswith("half__") else base
        support = self._consume(scope_name)
        made = self._materialise("diced__", whole, support, limit=1)
        return self._result(bool(made), "dice", args,
                            f"diced {name} into {made}" if made else
                            f"{name} has no diced__ product in this activity's scope")

    # ------------------------------------------------------------------ #
    def end_task(self) -> ToolResult:
        gs = self.world.goal_status()
        return ToolResult(ok=self.world.is_success(), tool="end_task", args={},
                          reason=f"goal_status={gs}", observation=self.observe())


# --------------------------------------------------------------------------- #
# benchmark audit
# --------------------------------------------------------------------------- #
def audit(activities=None) -> dict:
    """Classify every activity by whether the 17 tools can reach its goal.

    ``solvable`` -- every goal predicate is in ``AFFECTABLE_PREDICATES`` AND the
    activity's start state does not already satisfy the goal. ``blocked`` records
    the predicates that made it unreachable, so exclusions are auditable rather
    than a hand-maintained list.
    """
    from bddl.activity import get_all_activities

    if activities is None:
        activities = sorted(get_all_activities())
    solvable, blocked, contaminated, errors = [], {}, [], {}
    for act in activities:
        try:
            world = SymbolicWorld(act)
            preds = world.goal_predicates()
            missing = sorted(preds - AFFECTABLE_PREDICATES)
            if missing:
                blocked[act] = missing
            elif world.is_success():
                contaminated.append(act)
            else:
                solvable.append(act)
        except Exception as ex:                                    # noqa: BLE001
            errors[act] = f"{type(ex).__name__}: {ex}"
    return {"solvable": solvable, "blocked": blocked,
            "contaminated": contaminated, "errors": errors,
            "n_total": len(activities)}


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("activity", nargs="?", help="print one activity's start state")
    ap.add_argument("--audit", action="store_true",
                    help="classify all 1016 activities; writes --out if given")
    ap.add_argument("--out", help="write the solvable activity list as JSON")
    args = ap.parse_args()

    if args.audit:
        report = audit()
        n = report["n_total"]
        print(f"activities            : {n}")
        print(f"solvable              : {len(report['solvable'])}")
        print(f"already-satisfied     : {len(report['contaminated'])}")
        print(f"blocked (unsupported) : {len(report['blocked'])}")
        print(f"errors                : {len(report['errors'])}")
        counts: dict = {}
        for preds in report["blocked"].values():
            for p in preds:
                counts[p] = counts.get(p, 0) + 1
        print("\nblocking predicates:")
        for p, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {p:12s} {c:4d}")
        for act, err in list(report["errors"].items())[:10]:
            print(f"  ERROR {act}: {err}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
            print(f"\nwrote {args.out}")
        return

    act = args.activity or "picking_up_trash"
    world = SymbolicWorld(act)
    aci = SymbolicACI(world)
    print(f"activity     : {act}")
    print(f"goal preds   : {sorted(world.goal_predicates())}")
    print(f"goal status  : {world.goal_status()}")
    print(f"observation  : {json.dumps(aci.observe(), indent=1)}")


if __name__ == "__main__":
    _main()
