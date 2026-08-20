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
EDITS_TASK = [
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
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    applied = skipped = missing = 0

    for path, edits in ((args.file, EDITS), (args.task_file, EDITS_TASK)):
        src = open(path).read()
        if not args.check and not os.path.exists(path + ".orig"):
            shutil.copy(path, path + ".orig")
        changed = False
        for name, old, new in edits:
            if new in src:
                print(f"  [already] {name}")
                skipped += 1
            elif old in src:
                src = src.replace(old, new, 1)
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
