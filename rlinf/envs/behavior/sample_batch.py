"""Batch-drive `instance_generator` over many activities, and MEASURE the outcome.

Sampling is embarrassingly parallel across activities but each attempt must be its own
process -- OmniGibson locks `HEADLESS` at first boot and a second session in the same
process mutates the first one's scene.

Two things here are not conveniences, they are required by
`code-reading/BEHAVIOR task instance sampling & validation.md` §8.4:

**A crash watchdog.** Kit can die ~19 s into startup with
`carb.graphics-vulkan.plugin: GPU crash is detected` **and then hang** rather than
exit. Without a watchdog each crash costs the full timeout (measured: 45 min instead of
20 s). We tail the log and kill on the signature.

**Per-attempt GPU choice.** The crash rate tracks contention on the *chosen* card;
pinning a whole batch to one busy GPU turned an intermittent failure into a
deterministic one (28 of 28). We pick the least-loaded card per attempt.

We pin with `CUDA_VISIBLE_DEVICES` and do NOT also set `OMNIGIBSON_GPU_ID`. Swapping
those two while dropping the mask is a recorded day-loser: Kit then enumerates every
device, `createDevice()` fails, and the process segfaults before sampling begins.

Output is one JSONL row per attempt, which is the point -- the hit rate and the crash
rate are the numbers that decide whether scaling to ~1K activities is affordable.

    python -m rlinf.envs.behavior.sample_batch \
        --activities picking_up_trash,turning_on_radio --scene house_single_floor \
        --out /data/behavior-data/_samp0820 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

CRASH_SIGNATURES = (
    "GPU crash is detected",
    "eDeviceLost",              # the SECOND spelling of device loss; grepping only
    "Segmentation fault",       # the first scores dead runs as clean
    "Fatal Python error",
)

# ...but `eDeviceLost` ALSO appears, harmlessly, on every healthy headless boot:
#
#   [Warning] [carb.audio.output] failed to retrieve the capabilities for device 0
#             {result = eDeviceLost (2)}
#
# There is no sound card in the container. Matching the bare substring killed 12 of 12
# healthy sampling runs at ~14 s each and reported a 100% "crash rate" -- the exact
# mirror of the trap the docs warn about, and a reminder that a watchdog which is wrong
# in the FALSE-POSITIVE direction fabricates a measurement rather than losing one.
CRASH_EXCLUDE = ("carb.audio",)


def _crashed(tail: str) -> str | None:
    """Return the matched crash signature, ignoring known-benign lines."""
    for line in tail.splitlines():
        if any(x in line for x in CRASH_EXCLUDE):
            continue
        for sig in CRASH_SIGNATURES:
            if sig in line:
                return sig
    return None


def least_loaded_gpu(exclude: set[int]) -> int:
    """GPU index with the most free memory, ignoring ones we are already using."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, used = (int(x) for x in line.split(","))
        if idx not in exclude:
            rows.append((used, idx))
    if not rows:                       # every card already busy with our own attempts
        return 0
    return min(rows)[1]


SAMPLING_DIR = ("/opt/venv/openvla-oft/BEHAVIOR-1K/OmniGibson/omnigibson/sampling")


def headless_copy(script: str) -> str:
    """Copy an upstream sampling script with `gm.HEADLESS` forced True.

    The upstream scripts hardcode `gm.HEADLESS = False` at module scope, which cannot
    work in a container with no display -- Kit dies in `og.Environment(...)` and then
    hangs. We rewrite that one line into a sibling copy rather than editing the
    installed file, so the change is reproducible and lives nowhere but here.
    """
    src = os.path.join(SAMPLING_DIR, script)
    dst = os.path.join(SAMPLING_DIR, script.replace(".py", "_headless.py"))
    text = open(src).read().replace(
        "\ngm.HEADLESS = False",
        "\ngm.HEADLESS = True  # rewritten by rlinf sample_batch: no display here")
    with open(dst, "w") as f:
        f.write(text)
    return os.path.basename(dst)


def run_one(activity: str, args, gpu: int) -> dict:
    log_path = os.path.join(args.log_dir, f"{activity}.log")
    cwd = args.cwd
    if args.mode == "multiply":
        # Upstream's stage 3: derive N instances from an existing template. This is the
        # supported way to get `-tro_state` instances, and it is what the official
        # dataset was built with. It must run from the sampling directory: the script
        # does `from utils import validate_task` and opens `task_custom_lists.json`
        # by relative path.
        cwd = SAMPLING_DIR
        cmd = ["python", "-u", headless_copy("multiply_b1k_tasks.py"),
               "--seed", str(args.seed),
               "--start_idx", str(args.start_idx), "--end_idx", str(args.end_idx),
               "--partial_save", "--activity", activity]
        if args.scene:
            cmd += ["--scene_model", args.scene]
    else:
        cmd = [
            "python", "-u", "-m", "rlinf.envs.behavior.instance_generator",
            "--config", args.config,
            "--activity", activity,
            "--output-format", args.output_format,
            "--start-idx", str(args.start_idx), "--end-idx", str(args.end_idx),
            "--output-dir", args.out,
            "--num-trials", str(args.num_trials),
        ]
        if args.scene:
            cmd += ["--scene", args.scene]

    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               TORCHDYNAMO_DISABLE="1",   # quat_distance is traced by dynamo otherwise
               # upstream's utils.py imports gspread at module scope for a Google-Sheets
               # progress logger we neither have credentials for nor want
               PYTHONPATH="/data/behavior-data/_stubs:" + os.environ.get("PYTHONPATH", ""))

    t0 = time.time()
    outcome, detail = "unknown", ""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=cwd, env=env, start_new_session=True)
        while True:
            rc = proc.poll()
            if rc is not None:
                outcome = "exit_%d" % rc
                break
            if time.time() - t0 > args.timeout:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                outcome, detail = "timeout", f"{args.timeout}s"
                break
            # watchdog: a crashed Kit hangs, so react to the log, not to the exit code
            try:
                tail = open(log_path, errors="ignore").read()[-8000:]
            except OSError:
                tail = ""
            hit = _crashed(tail)
            if hit:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                outcome, detail = "crash", hit
                break
            time.sleep(2.0)

    text = open(log_path, errors="ignore").read()
    if outcome.startswith("exit") and args.mode == "multiply":
        n_saved = text.count("trial 0 saved")
        if n_saved:
            outcome, detail = "saved", f"{n_saved} instances"
        elif "Did not find default stable scene json" in text:
            outcome, detail = "no_stable_scene", "run create_stable_scene first"
        elif "Traceback" in text:
            outcome = "error"
            detail = (re.search(r"\n(\w+Error: .{0,100})", text) or [None, ""])[1]
    elif outcome.startswith("exit"):
        if "saved " in text:
            outcome = "saved"
            detail = (re.search(r"saved (\S+)", text) or [None, ""])[1]
        elif "validation failed" in text:
            outcome = "validation_failed"
            detail = (re.search(r"validation failed: (.{0,120})", text) or [None, ""])[1]
        elif "sampling failed" in text:
            outcome = "sampling_failed"
            detail = (re.search(r"sampling failed: (.{0,120})", text) or [None, ""])[1]
        elif "Error during instance generation" in text:
            outcome = "error"
            detail = (re.search(r"Error during instance generation: (.{0,120})", text)
                      or [None, ""])[1]

    row = {"activity": activity, "gpu": gpu, "outcome": outcome,
           "detail": detail, "seconds": round(time.time() - t0, 1)}
    print(f"  {activity:<44} {outcome:<18} {row['seconds']:>7.1f}s  gpu{gpu}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activities", required=True,
                    help="comma-separated, or @file with one per line")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default="/data/behavior-data/_tmp/samp_logs")
    ap.add_argument("--config",
                    default="/workspace/RLinf/examples/embodiment/config/env/behavior_r1pro.yaml")
    ap.add_argument("--cwd", default="/workspace/RLinf")
    ap.add_argument("--output-format", default="tro_state")
    ap.add_argument("--start-idx", type=int, default=900)
    ap.add_argument("--end-idx", type=int, default=900)
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--mode", default="generate", choices=["generate", "multiply"],
                    help="generate: sample a task from scratch (our instance_generator). "
                         "multiply: derive instances from an existing template using "
                         "upstream's multiply_b1k_tasks.py -- the supported path, and "
                         "much more reliable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="parallel attempts")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per attempt")
    ap.add_argument("--jsonl", default="/data/behavior-data/_tmp/sample_batch.jsonl")
    args = ap.parse_args()

    spec = args.activities
    acts = ([a.strip() for a in open(spec[1:]) if a.strip()] if spec.startswith("@")
            else [a.strip() for a in spec.split(",") if a.strip()])
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    print(f"activities : {len(acts)}\njobs       : {args.jobs}\nscene      : {args.scene}",
          flush=True)

    in_flight: set[int] = set()
    rows = []

    def task(activity: str) -> dict:
        gpu = least_loaded_gpu(in_flight)
        in_flight.add(gpu)
        try:
            return run_one(activity, args, gpu)
        finally:
            in_flight.discard(gpu)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(task, acts):
            rows.append(row)
            with open(args.jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")

    n = len(rows)
    saved = sum(r["outcome"] == "saved" for r in rows)
    crash = sum(r["outcome"] == "crash" for r in rows)
    print(f"\n=== {n} attempts in {(time.time() - t0) / 60:.1f} min "
          f"({args.jobs} parallel) ===")
    print(f"  saved       : {saved}/{n} = {100.0 * saved / max(n, 1):.0f}%   "
          f"(§8.5 measured 43% on the old host)")
    print(f"  Kit crashes : {crash}/{n} = {100.0 * crash / max(n, 1):.0f}%   "
          f"(§8.4 measured ~50% on the old host)")
    by = {}
    for r in rows:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20} {v}")
    print(f"  rows -> {args.jsonl}")


if __name__ == "__main__":
    main()
