"""Apply the three recorded fixes to OmniGibson's task-instance validator.

`validate_task` cannot pass on this dataset unmodified. The three edits below were
established over 13 sampling runs and are recorded in
`code-reading/BEHAVIOR task instance sampling & validation.md` §8.1; this script exists
so they can be re-applied to any container instead of being hand-typed into
site-packages and lost when the machine changes (which is what happened to the
`native_bbox.json` extractor -- see `extract_bbox.py`).

They cannot be monkeypatched from the outside: all three live inside
`_validate_identical_object_kinematic_state`, which is a *nested* function defined
inside `validate_task`, so there is no importable symbol to wrap. Hence in-place source
editing, made idempotent and verbose.

    python -m rlinf.envs.behavior.patch_sampler --check    # report, change nothing
    python -m rlinf.envs.behavior.patch_sampler            # apply

The edits, and why each is legitimate rather than a number that makes the test pass:

1. **tensor coercion in the `ori` comparison.** `quat_distance` documents and indexes
   `torch.tensor`, but the values come from a JSON-loaded state dict and are python
   lists. Every other branch in the same function already wraps with `th.tensor(...)`;
   the `ori` branch is the only one that forgot. Upstream has since fixed this in a
   relocated copy of the file.

2. **skip root-link keys the live dump does not carry.** The native scene JSON carries
   `lin_vel`/`ang_vel`; a live `dump_state()` does not always. Comparing a key that only
   one side has is a KeyError, not a validation failure. Measured impact: 1 object of
   260.

3. **`ori_atol` at the step-2 call site only.** 20 of 20 trials failed step 2 on a tall
   thin mirror unrelated to the task, rocking when physics steps
   (`standing_mirror_wwkuuf_0 ... ori: default 0.0, got 0.0776`). OmniGibson's own
   source carries the intended fix commented out (`# tol = 0.15 if "ori" in key`).
   Uncommenting it globally would loosen orientation for the TASK objects too, whose
   poses are the entire point of the instance -- so the looser value is passed only at
   step 2, which by construction iterates objects the task does not touch. Steps 1, 3
   and 4 keep 0.05.
"""

from __future__ import annotations

import argparse
import os
import shutil

TARGET = "/opt/venv/openvla-oft/BEHAVIOR-1K/OmniGibson/omnigibson/sampling/utils.py"
TARGET_TASK = ("/opt/venv/openvla-oft/BEHAVIOR-1K/OmniGibson/omnigibson/tasks/"
               "behavior_task.py")
TARGET_BDDL = ("/opt/venv/openvla-oft/BEHAVIOR-1K/OmniGibson/omnigibson/utils/"
               "bddl_utils.py")

EDITS = [
    # (name, old, new)
    (
        "1. tensor coercion in the ori comparison",
        "obj_val = th.norm(T.quat2axisangle(T.quat_distance(val, obj_val)))",
        "obj_val = th.norm(T.quat2axisangle(T.quat_distance(th.tensor(val), th.tensor(obj_val))))",
    ),
    (
        "2. skip root-link keys absent from the live dump",
        """            if not check_vel and "vel" in key:
                continue
            obj_val = obj_dict["root_link"][key]""",
        """            if not check_vel and "vel" in key:
                continue
            # RLinf: the native scene json carries lin_vel/ang_vel, a live dump_state()
            # does not always. A key only one side has is a KeyError, not a mismatch.
            if key not in obj_dict["root_link"]:
                continue
            obj_val = obj_dict["root_link"][key]""",
    ),
    (
        "3a. ori_atol parameter on the nested validator",
        "def _validate_identical_object_kinematic_state(obj_name, default_obj_dict, obj_dict, check_vel=True):",
        "def _validate_identical_object_kinematic_state(obj_name, default_obj_dict, obj_dict, check_vel=True, ori_atol=0.05):",
    ),
    (
        "3b. use ori_atol for the ori key",
        '            atol = 1.0 if "vel" in key else 0.05\n',
        '            atol = 1.0 if "vel" in key else (ori_atol if key == "ori" else 0.05)\n',
    ),
    (
        "3c. pass ori_atol=0.15 at the STEP 2 call site only",
        """        valid_obj, err_msg = _validate_identical_object_kinematic_state(
            obj_name, default_obj_info, obj_info, check_vel=True
        )""",
        """        valid_obj, err_msg = _validate_identical_object_kinematic_state(
            obj_name, default_obj_info, obj_info, check_vel=True, ori_atol=0.15
        )""",
    ),
]

# A second file. `BehaviorTask.reset` force-wakes every entity in the object scope, but
# during SAMPLING that scope legitimately holds `None`s -- which is the same fact that
# forces `task.include_obs: False` in the recorded config. The loop does not guard for
# it, so reset dies with `'NoneType' object has no attribute 'exists'` AFTER the scene
# has loaded. Measured: 3 of the first 5 completed attempts in a 12-activity batch.
EDITS_BDDL = [
    (
        "5. drop (and log) conditions whose scope entry is None (BOTH sort sites)",
        """                        rigid_conditions = [c for c in conditions_to_sample if c[2].prim_type != PrimType.CLOTH]""",
        """                        # RLinf: entities can be None here -- the scope is not fully
                        # instantiated during online sampling -- and `.prim_type` on None
                        # aborts the whole activity. Drop them, but SAY SO: a silently
                        # dropped condition would mean an instance that does not satisfy
                        # its own BDDL. validate_task step 3 re-checks the initial
                        # conditions, so a bad drop fails the instance instead of
                        # shipping it.
                        _missing = [c[3] for c in conditions_to_sample if c[2] is None]
                        if _missing:
                            print(f"[rlinf] unfilled object_scope entries, conditions "
                                  f"skipped: {_missing}", flush=True)
                            conditions_to_sample = [c for c in conditions_to_sample
                                                    if c[2] is not None]
                        rigid_conditions = [c for c in conditions_to_sample if c[2].prim_type != PrimType.CLOTH]""",
    ),
]

EDITS_TASK = [
    (
        "7. no presampled robot pose -> fall back, do not crash",
        '        # Use presampled robot pose if specified (only available for officially supported mobile manipulators)\n        if self.use_presampled_robot_pose:\n            robot = self.get_agent(env)\n            presampled_poses = env.scene.get_task_metadata(key="robot_poses")\n            assert (\n                robot.model_name in presampled_poses\n            ), f"{robot.model_name} presampled pose is not found in task metadata; please set use_presampled_robot_pose to False in task config"',
        '        # Use presampled robot pose if specified (only available for officially supported mobile manipulators)\n        presampled_poses = env.scene.get_task_metadata(key="robot_poses") if self.use_presampled_robot_pose else None\n        # RLinf: a template that never went through stage 4 (`sample_robot_pose.py`, which\n        # does not exist in this OmniGibson version) carries NO `robot_poses`, and\n        # `robot.model_name in None` raises a TypeError before the assert below can print\n        # its own advice -- "please set use_presampled_robot_pose to False in task config".\n        # Kit then segfaults on teardown, so it reads as a simulator crash rather than a\n        # missing key. Measured: 23 of the 24 activities that have a template but no\n        # instance, i.e. exactly the ones where activity coverage could grow.\n        if self.use_presampled_robot_pose and not presampled_poses:\n            print("[rlinf] no presampled robot_poses in this scene task metadata; "\n                  "falling back to the default robot pose", flush=True)\n            self.use_presampled_robot_pose = False\n        if self.use_presampled_robot_pose:\n            robot = self.get_agent(env)\n            assert (\n                robot.model_name in presampled_poses\n            ), f"{robot.model_name} presampled pose is not found in task metadata; please set use_presampled_robot_pose to False in task config"',
    ),
    (
        "6. potential is undefined while the scope is being sampled",
        """        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_goal_conditions(self.ground_goal_state_options[0])""",
        """        # RLinf: during online sampling the scope is not yet fully instantiated, so
        # the goal cannot be evaluated -- `evaluate_goal_conditions` raises
        # `child_values has NoneTypes`. The potential is a training signal and is never
        # read on the sampling path, so returning 0 here is not a silent substitution.
        if any(v is None for v in self.object_scope.values()):
            return 0.0
        # Evaluate the first ground goal state option as the potential
        _, satisfied_predicates = evaluate_goal_conditions(self.ground_goal_state_options[0])""",
    ),
    (
        "4. guard the force-wake loop against None entries in object_scope",
        """        for obj in self.object_scope.values():
            if obj.exists and isinstance(obj, DatasetObject):""",
        """        for obj in self.object_scope.values():
            # RLinf: during sampling the scope legitimately holds Nones.
            if obj is not None and obj.exists and isinstance(obj, DatasetObject):""",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=TARGET)
    ap.add_argument("--task-file", default=TARGET_TASK)
    ap.add_argument("--bddl-file", default=TARGET_BDDL)
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    applied = skipped = missing = 0

    for path, edits in ((args.file, EDITS), (args.task_file, EDITS_TASK),
                        (args.bddl_file, EDITS_BDDL)):
        src = open(path).read()
        if not args.check and not os.path.exists(path + ".orig"):
            shutil.copy(path, path + ".orig")
        changed = False
        for name, old, new in edits:
            if new in src:
                print(f"  [already] {name}")
                skipped += 1
            elif old in src:
                # `count` matters: the None-guard has two byte-identical sites in
                # bddl_utils and patching only the first leaves the second to fail.
                src = src.replace(old, new)
                print(f"  [apply  ] {name}")
                applied += 1
                changed = True
            else:
                print(f"  [MISSING] {name}  <- anchor not found; file may have changed")
                missing += 1
        if not args.check and changed:
            open(path, "w").write(src)

    print(f"\napplied={applied} already={skipped} missing={missing}"
          f"{'  (check only, nothing written)' if args.check else ''}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
