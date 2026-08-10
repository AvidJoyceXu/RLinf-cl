#!/usr/bin/env python3
"""Parse the sequential-SFT (LoRA) OpenVLA-OFT eval logs into a per-task success matrix.

The raw logs report a *cumulative* "# successes: N (X%)" counter across the whole
file, so per-task rates have to be recovered by counting the `Success:` lines
inside each `Task:` block instead of reading that counter.

Usage:
    python rebuttal/parse_sft_eval_logs.py
"""

import glob
import os
import re
from collections import OrderedDict

LOG_ROOT = os.path.join(os.path.dirname(__file__), "openvla测试数据")


def parse_log(path):
    """Return {task_description: (num_success, num_episodes)} for one eval log."""
    counts = OrderedDict()
    current = None
    for line in open(path):
        task_match = re.match(r"^Task: (.+)$", line)
        if task_match:
            current = task_match.group(1).strip()
            counts.setdefault(current, [0, 0])
            continue
        success_match = re.match(r"^Success: (True|False)", line)
        if success_match and current is not None:
            counts[current][1] += 1
            if success_match.group(1) == "True":
                counts[current][0] += 1
    return OrderedDict((k, tuple(v)) for k, v in counts.items())


def main():
    for suite in ("libero_object", "libero_spatial"):
        rows = OrderedDict()
        tasks = []
        for path in sorted(glob.glob(os.path.join(LOG_ROOT, suite, "*.txt"))):
            stage = re.search(r"policy_after_(task\d+)", path).group(1)
            per_task = parse_log(path)
            if not tasks:
                tasks = list(per_task.keys())
            rows[stage] = per_task

        print("=" * 30, suite)
        print(f"{'policy':<10}" + "".join(f"{t.split()[3][:14]:>16}" for t in tasks) + f"{'avg':>8}")
        for stage, per_task in rows.items():
            rates = [100.0 * per_task[t][0] / per_task[t][1] for t in tasks]
            print(
                f"{stage:<10}"
                + "".join(f"{r:>15.1f}%" for r in rates)
                + f"{sum(rates) / len(rates):>7.1f}%"
            )
        n_eps = {t: rows[next(iter(rows))][t][1] for t in tasks}
        print(f"episodes per task: {sorted(set(n_eps.values()))}")
        print()


if __name__ == "__main__":
    main()
