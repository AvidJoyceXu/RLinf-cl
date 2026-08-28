"""Policy-legal reference replay for the frozen text-detect benchmark.

The planner may use privileged identity and geometry to decide where to search, but
the emitted trajectory contains only public tools.  Every object-selection argument
is a ``dN`` handle present in the immediately preceding policy observation; every
view or world change forces a new observation.  This is the sequential-solvability
certificate and later the source of matched detect SFT demonstrations.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from rlinf.envs.behavior.symbolic_expert import _plan_atom, _split
from rlinf.envs.behavior.symbolic_world import (
    SymbolicACI,
    SymbolicWorld,
    lemma_of,
)


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class ReferenceFailure(RuntimeError):
    """One mutually exclusive failure for a reference replay."""

    def __init__(self, failure_class: str, detail: str):
        super().__init__(detail)
        self.failure_class = failure_class
        self.detail = detail


class DetectReference:
    """Privileged route choice behind a strictly public handle/action trace."""

    def __init__(
        self,
        row: dict,
        max_turns: int = 2000,
        obs_mode: str = "detect_scope",
    ):
        self.row = row
        self.max_turns = max_turns
        source = row["instance_source"]
        self.world = SymbolicWorld(
            row["activity"],
            instance_source_selection=source,
        )
        self.aci = SymbolicACI(
            self.world,
            obs_mode=obs_mode,
            selection_mode="handle",
            layout_source_selection=source,
        )
        actual = Path(self.aci.layout.instance_path).name
        expected = row["geometry_instance_key"]
        if actual != expected:
            raise ReferenceFailure(
                "manifest_mismatch",
                f"resolved geometry {actual!r}, expected {expected!r}",
            )
        self.trace: list[dict] = []
        self.handle_resolutions: list[dict] = []
        self.first_visible: dict[str, int] = {}
        self.required_objects: set[str] = set()
        self.rejections: list[str] = []
        self.goal_options = self.world.ground_goal_options()
        self._selected_scope: str | None = None
        self._scope_alias = {name: name for name in self.world.scope_names}
        self._bound_roles: set[str] = set()

    @property
    def turns(self) -> int:
        return len(self.trace)

    def _check_budget(self) -> None:
        if self.turns >= self.max_turns:
            raise ReferenceFailure(
                "budget_exhausted",
                f"reference exceeded {self.max_turns} public tool turns",
            )

    def _observe(self):
        self._check_budget()
        observation = self.aci.observe()
        self.trace.append(
            {
                "name": "observe",
                "arguments": {},
                "ok": True,
                "result": observation,
                "audit": {
                    "view_xy": [self.aci.view.x, self.aci.view.y],
                    "epoch": list(self.aci._observation_epoch()),
                    "scope_detections": [
                        {"handle": item.det, "scope": item.key[len("scope:") :]}
                        for item in self.aci._dets
                        if item.key.startswith("scope:")
                    ],
                },
            }
        )
        for detection in self.aci._dets:
            if detection.key.startswith("scope:"):
                name = detection.key[len("scope:") :]
                self.first_visible.setdefault(name, self.turns)
        return observation

    def _action(self, name: str, arguments: dict):
        self._check_budget()
        result = getattr(self.aci, name)(**arguments)
        step = {
            "name": name,
            "arguments": arguments,
            "ok": bool(result.ok),
            "result": (
                result.observation
                if result.observation is not None
                else {"ok": bool(result.ok), "reason": result.reason}
            ),
        }
        if result.observation is not None:
            step["audit"] = {
                "view_xy": [self.aci.view.x, self.aci.view.y],
                "epoch": list(self.aci._observation_epoch()),
                "scope_detections": [
                    {"handle": item.det, "scope": item.key[len("scope:") :]}
                    for item in self.aci._dets
                    if item.key.startswith("scope:")
                ],
            }
            for detection in self.aci._dets:
                if detection.key.startswith("scope:"):
                    scope = detection.key[len("scope:") :]
                    self.first_visible.setdefault(scope, self.turns + 1)
        self.trace.append(step)
        if not result.ok:
            self.rejections.append(f"{name}({arguments}) -> {result.reason}")
            raise ReferenceFailure("tool_rejection", self.rejections[-1])
        return result

    def _current_handle(self, scope_name: str) -> str | None:
        key = f"scope:{scope_name}"
        detection = next((item for item in self.aci._dets if item.key == key), None)
        return detection.det if detection is not None else None

    def _canonical_public_handle(self, target: str, tool: str) -> str | None:
        """Choose the first public handle among goal-equivalent instances.

        Category, state and handle order come from the current public view.  Hidden
        The stable ``sN`` track makes the binding persistent across views; hidden
        scope names are retained only in certificate audit metadata. Activities for
        which this public binding cannot complete BDDL are excluded rather than
        repaired with a target-aware choice.
        """
        if not self.trace:
            return None
        step = self.trace[-1]
        detections = (step.get("result") or {}).get("detections", [])
        handle_to_scope = {
            str(item["handle"]): str(item["scope"])
            for item in (step.get("audit") or {}).get("scope_detections", [])
        }
        category = lemma_of(target)

        def action_compatible(detection: dict) -> bool:
            states = detection.get("states") or {}
            if tool == "open":
                return states.get("Open") is False
            if tool == "close":
                return states.get("Open") is True
            if tool == "place_inside" and "Open" in states:
                return states["Open"] is True
            return True

        candidates = []
        bound_actuals = {self._scope_alias[role] for role in self._bound_roles}
        for detection in detections:
            handle = str(detection.get("det", ""))
            scope = handle_to_scope.get(handle)
            if (
                scope is None
                or scope in bound_actuals
                or detection.get("category") != category
                or not action_compatible(detection)
            ):
                continue
            candidates.append((int(handle[1:]), handle, scope))
        if not candidates:
            return None
        _, handle, scope = min(candidates)
        self._selected_scope = scope
        return handle

    def _camera(self, name: str, target: str) -> str | None:
        self._action(name, {})
        self._observe()
        return self._current_handle(target)

    def _move_to_target_ring(
        self,
        target: str,
        radius: float,
        bearing_offset: float = 0.0,
    ) -> str | None:
        """Reach a target-facing stand using only move/strafe primitives."""
        bbox = self.aci._observation_bbox(target)
        if bbox is None:
            raise ReferenceFailure(
                "missing_dynamic_geometry",
                f"{target} has no current observation bbox",
            )
        centre, _ = bbox
        yaw = float(self.aci.view.yaw)
        forward = (math.cos(yaw), math.sin(yaw))
        left = (-forward[1], forward[0])
        approach = (
            math.cos(yaw + bearing_offset),
            math.sin(yaw + bearing_offset),
        )
        desired = (
            float(centre[0]) - radius * approach[0],
            float(centre[1]) - radius * approach[1],
        )
        delta = (desired[0] - self.aci.view.x, desired[1] - self.aci.view.y)
        forward_steps = round((delta[0] * forward[0] + delta[1] * forward[1]) / 0.5)
        left_steps = round((delta[0] * left[0] + delta[1] * left[1]) / 0.5)
        if abs(forward_steps) > 100 or abs(left_steps) > 100:
            raise ReferenceFailure(
                "route_out_of_bounds",
                f"target route needs {forward_steps=} {left_steps=}",
            )
        moved_since_observe = 0

        def move(tool: str) -> str | None:
            nonlocal moved_since_observe
            self._action(tool, {})
            moved_since_observe += 1
            if moved_since_observe < 4:
                return None
            moved_since_observe = 0
            self._observe()
            return self._current_handle(target)

        forward_tool = "move_ahead" if forward_steps >= 0 else "move_back"
        for _ in range(abs(forward_steps)):
            handle = move(forward_tool)
            if handle is not None:
                return handle
        lateral_tool = "strafe_left" if left_steps >= 0 else "strafe_right"
        for _ in range(abs(left_steps)):
            handle = move(lateral_tool)
            if handle is not None:
                return handle
        # Privileged route choice does not pretend to be fully reactive search.
        # Check every four 0.5 m moves (plus the destination): this can stop after an
        # object becomes selectable without doubling oracle cost by observing every
        # blind step.
        if moved_since_observe or (not forward_steps and not left_steps):
            self._observe()
        return self._current_handle(target)

    def _pitch_search(self, target: str) -> str | None:
        """Search the public pitch range and leave every view explicitly observed."""
        while self.aci.view.pitch < -1e-6:
            handle = self._camera("look_up", target)
            if handle is not None:
                return handle
        while self.aci.view.pitch > 1e-6:
            handle = self._camera("look_down", target)
            if handle is not None:
                return handle
        for tool in (
            "look_down",
            "look_down",
            "look_up",
            "look_up",
            "look_up",
            "look_up",
            "look_down",
            "look_down",
        ):
            handle = self._camera(tool, target)
            if handle is not None:
                return handle
        return None

    def acquire(self, target: str, tool: str, *, canonical: bool = False) -> str:
        """Return a live handle from the immediately preceding observation."""
        self.required_objects.add(target)
        self._observe()
        handle = (
            self._canonical_public_handle(target, tool)
            if canonical
            else self._current_handle(target)
        )
        if handle is not None:
            return handle
        support = self.world.support_of(target)
        if support is not None:
            parent = support[1]
            parent_handle = self._current_handle(parent)
            if parent_handle is not None:
                # Re-navigate to an already visible open host. `go_to` is oracle
                # navigation and chooses a target-visible approach; the following
                # public observation still supplies the only legal child handle.
                parent_role = next(
                    role
                    for role, actual in self._scope_alias.items()
                    if actual == parent
                )
                self._selected_action("go_to", parent_role)
                self._observe()
                handle = (
                    self._canonical_public_handle(target, tool)
                    if canonical
                    else self._current_handle(target)
                )
                if handle is not None:
                    return handle
        bbox = self.aci._observation_bbox(target)
        if bbox is None:
            raise ReferenceFailure(
                "target_not_observable",
                f"{target} has no current bbox (closed, held, consumed, or unsized)",
            )
        if self.aci.obs_mode in self.aci.SCAN_MODES:
            for _ in range(len(self.aci._scan_candidates)):
                self._action("scan_next", {})
                handle = (
                    self._canonical_public_handle(target, tool)
                    if canonical
                    else self._current_handle(target)
                )
                if handle is not None:
                    return handle
            raise ReferenceFailure(
                "target_not_selectable",
                f"{target} was not detected in one complete public scan cycle",
            )
        _, extent = bbox
        base_radius = max(1.2, math.hypot(extent[0], extent[1]) + 0.6)
        for ring_offset in (0.0, 1.0, 2.0):
            for _ in range(4):
                # Public yaw changes are 90 degrees, but solvable views can lie
                # between those four axes. Offset the stand point while keeping the
                # target within the camera's ±35-degree horizontal half-FOV. Fifteen
                # degree spacing covers the full orbit with margin for 0.5 m motion
                # quantization.
                for degrees in (0, 15, -15, 30, -30):
                    bearing_offset = math.radians(degrees)
                    handle = self._move_to_target_ring(
                        target,
                        base_radius + ring_offset,
                        bearing_offset,
                    )
                    if handle is None:
                        handle = self._pitch_search(target)
                    if handle is not None:
                        return handle
                handle = self._camera("turn_left", target)
                if handle is not None:
                    return handle
        raise ReferenceFailure(
            "target_not_selectable",
            f"{target} was not detected from any primitive-reachable target orbit",
        )

    def _selected_action(
        self,
        tool: str,
        target_role: str,
        *,
        argument_name: str = "name",
        extra: dict | None = None,
    ):
        target = self._scope_alias[target_role]
        self._selected_scope = target
        canonical = (
            self.aci.obs_mode in self.aci.SCAN_MODES
            and target_role not in self._bound_roles
        )
        handle = self.acquire(target, tool, canonical=canonical)
        if not canonical:
            # A support-reacquisition inside `acquire` may execute its own selector
            # and overwrite the audit scratch field. The returned handle still
            # resolves the requested bound target; restore that scope explicitly.
            self._selected_scope = target
        if canonical:
            selected = self._selected_scope
            if selected is None:
                raise ReferenceFailure(
                    "illegal_selection",
                    f"canonical selector for {target_role} has no scope resolution",
                )
            if selected != target:
                peer_role = next(
                    role
                    for role, actual in self._scope_alias.items()
                    if actual == selected
                )
                self._scope_alias[target_role], self._scope_alias[peer_role] = (
                    selected,
                    target,
                )
            self._bound_roles.add(target_role)
            target = self._scope_alias[target_role]
            self._selected_scope = target
        if self.trace[-1]["name"] not in ("observe", "scan_next"):
            raise ReferenceFailure(
                "illegal_selection",
                f"{tool} selector was not preceded by a public detection view",
            )
        if self.aci._det_epoch != self.aci._observation_epoch():
            raise ReferenceFailure("illegal_selection", f"stale selector before {tool}")
        arguments = {argument_name: handle, **(extra or {})}
        selected_public = next(
            (
                detection
                for detection in (self.trace[-1].get("result") or {}).get(
                    "detections", []
                )
                if detection.get("det") == handle
            ),
            {},
        )
        self.handle_resolutions.append(
            {
                "turn": self.turns + 1,
                "tool": tool,
                "argument": argument_name,
                "handle": handle,
                "scope": self._selected_scope,
                "planned_scope": target_role,
                "track": selected_public.get("track"),
                "binding_created": canonical,
                "epoch": list(self.aci._observation_epoch()),
            }
        )
        return self._action(tool, arguments)

    def _scope(self, display_name: str) -> str:
        scope_name = self.world.from_display.get(display_name, display_name)
        if scope_name not in self.world.scope:
            raise ReferenceFailure(
                "planner_identity_error",
                f"planner emitted unknown object {display_name!r}",
            )
        return scope_name

    def _execute_planned_step(self, tool: str, arguments: dict) -> None:
        if tool == "go_to":
            self._selected_action(tool, self._scope(arguments["name"]))
            return
        if tool in {
            "grasp",
            "open",
            "close",
            "toggle_on",
            "toggle_off",
            "cook",
            "slice",
            "dice",
        }:
            self._selected_action(tool, self._scope(arguments["name"]))
            return
        if tool in {"spray", "uncover", "fill"}:
            system = self._scope(arguments["system_name"])
            self._selected_action(
                tool,
                self._scope(arguments["name"]),
                extra={"system_name": lemma_of(system)},
            )
            return
        if tool in {"place_on", "place_inside"}:
            target_key = "surface" if tool == "place_on" else "container"
            self._selected_action(
                tool,
                self._scope(arguments[target_key]),
                argument_name=target_key,
                extra={"name": "held"},
            )
            return
        if tool == "release":
            self._action("release", {"name": "held"})
            return
        raise ReferenceFailure("unsupported_planner_tool", tool)

    def run(self) -> dict:
        failure = None
        try:
            atoms = [_split(atom) for atom in self.world.ground_goal_atoms()]
            atoms.sort(key=lambda item: item[0], reverse=True)
            for positive, predicate, arguments in atoms:
                # Canonical public choices may permute goal-equivalent instances.
                # Test progress through that public binding, while leaving the
                # planner arguments in their original goal-role namespace so
                # `_selected_action` applies the same binding exactly once.
                bound_arguments = [
                    self._scope_alias.get(argument, argument) for argument in arguments
                ]
                if self.world.holds(predicate, bound_arguments) == positive:
                    continue
                plan = _plan_atom(self.world, positive, predicate, arguments)
                if not plan:
                    if predicate == "real" and not positive:
                        continue
                    raise ReferenceFailure(
                        "unplanned_atom",
                        ("" if positive else "not ")
                        + f"{predicate}({', '.join(arguments)})",
                    )
                for tool, tool_arguments in plan:
                    self._execute_planned_step(tool, tool_arguments)
            success = self.world.is_success()
            if not success:
                raise ReferenceFailure(
                    "goal_unsatisfied", str(self.world.goal_status())
                )
            self._action("end_task", {})
        except ReferenceFailure as error:
            failure = {"class": error.failure_class, "detail": error.detail}

        success = self.world.is_success()
        ended = bool(self.trace and self.trace[-1]["name"] == "end_task")
        first_selectable = {}
        for item in self.handle_resolutions:
            first_selectable.setdefault(item["scope"], item["turn"])
        return {
            "activity": self.row["activity"],
            "split": self.row["split"],
            "instance_source": self.row["instance_source"],
            "geometry_instance_key": self.row["geometry_instance_key"],
            "obs_mode": self.aci.obs_mode,
            "goal_satisfied": success,
            "proper_completion": success and ended,
            "policy_legal": failure is None,
            "turns": self.turns,
            "tool_counts": dict(
                sorted(
                    (name, sum(step["name"] == name for step in self.trace))
                    for name in {step["name"] for step in self.trace}
                )
            ),
            "required_objects": len(self.required_objects),
            "required_visible": sum(
                name in self.first_visible for name in self.required_objects
            ),
            "required_selectable": sum(
                name in first_selectable for name in self.required_objects
            ),
            "first_visible": self.first_visible,
            "first_selectable": first_selectable,
            "handle_resolutions": self.handle_resolutions,
            "rejections": self.rejections,
            "failure": failure,
            "trace": self.trace,
        }


def run_manifest(
    manifest: dict,
    max_turns: int,
    trace_dir: str | None = None,
    obs_mode: str = "detect_scope",
) -> dict:
    results = []
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(manifest["rows"], 1):
        try:
            result = DetectReference(
                row,
                max_turns=max_turns,
                obs_mode=obs_mode,
            ).run()
        except Exception as error:  # noqa: BLE001 - classified harness construction fail
            result = {
                "activity": row["activity"],
                "split": row["split"],
                "instance_source": row["instance_source"],
                "goal_satisfied": False,
                "proper_completion": False,
                "policy_legal": False,
                "turns": 0,
                "failure": {
                    "class": "construction_error",
                    "detail": f"{type(error).__name__}: {error}",
                },
                "trace": [],
            }
        if trace_dir:
            trace = result.pop("trace")
            trace_path = Path(trace_dir) / f"{row['split']}__{row['activity']}.json"
            with trace_path.open("w") as stream:
                json.dump({"summary": result, "trace": trace}, stream, indent=2)
            result["trace_file"] = trace_path.name
        else:
            result.pop("trace", None)
        results.append(result)
        if index % 10 == 0:
            solved = sum(item["proper_completion"] for item in results)
            print(f"[{index}/{len(manifest['rows'])}] proper={solved}", flush=True)
    certified = [item for item in results if item["proper_completion"]]
    failures = collections.Counter(
        item["failure"]["class"] for item in results if item.get("failure") is not None
    )
    certificate = {
        "schema": (
            "text-detect-policy-identifiable-reference-v2"
            if obs_mode == "detect_scope_scan"
            else "text-detect-policy-legal-reference-v1"
        ),
        "obs_mode": obs_mode,
        "input_manifest_content_sha256": manifest["content_sha256"],
        "max_turns": max_turns,
        "counts": {
            "rows": len(results),
            "goal_satisfied": sum(item["goal_satisfied"] for item in results),
            "proper_completion": sum(item["proper_completion"] for item in results),
            "policy_legal": sum(item["policy_legal"] for item in results),
            "certified_by_split": {
                split: sum(item["split"] == split for item in certified)
                for split in ("train", "s2")
            },
            "failure_classes": dict(sorted(failures.items())),
        },
        "certified_activities": [item["activity"] for item in certified],
        "oracle_turns": {item["activity"]: item["turns"] for item in certified},
        "results": results,
    }
    certificate["content_sha256"] = _canonical_sha256(certificate)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument(
        "--obs-mode",
        choices=("detect_scope", "detect_scope_scan"),
        default="detect_scope",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--activities",
        help="optional comma-separated manifest activities for targeted debugging",
    )
    args = parser.parse_args()
    with open(args.manifest) as stream:
        manifest = json.load(stream)
    if args.activities:
        selected = {name.strip() for name in args.activities.split(",") if name.strip()}
        manifest = {
            **manifest,
            "rows": [row for row in manifest["rows"] if row["activity"] in selected],
        }
        missing = selected - {row["activity"] for row in manifest["rows"]}
        if missing:
            parser.error(f"activities absent from manifest: {sorted(missing)}")
    if args.limit:
        manifest = {**manifest, "rows": manifest["rows"][: args.limit]}
    certificate = run_manifest(
        manifest,
        args.max_turns,
        args.trace_dir,
        obs_mode=args.obs_mode,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(certificate, stream, indent=2)
        stream.write("\n")
    print(json.dumps(certificate["counts"], sort_keys=True))
    print(f"content_sha256={certificate['content_sha256']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
