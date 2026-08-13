"""Dump the latest value of every scalar in a tensorboard run dir.

Usage: python rebuttal/read_tb.py <log_dir> [key_substring ...]
"""

import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    path = sys.argv[1]
    filters = sys.argv[2:]

    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags()["scalars"]
    if filters:
        tags = [t for t in tags if any(f in t for f in filters)]

    last_step = 0
    by_prefix = defaultdict(list)
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if not events:
            continue
        last_step = max(last_step, events[-1].step)
        prefix = tag.split("/")[0] if "/" in tag else "_"
        by_prefix[prefix].append((tag, events))

    print(f"# {path}")
    print(f"# last step: {last_step}   n_scalars: {len(tags)}")
    for prefix in sorted(by_prefix):
        print(f"\n[{prefix}]")
        for tag, events in by_prefix[prefix]:
            first, last = events[0], events[-1]
            print(
                f"  {tag:<55} step {last.step:>6}  last={last.value:.5f}  "
                f"(first step {first.step}: {first.value:.5f}, n={len(events)})"
            )

    # wall-clock throughput, measured on whichever scalar has the most points
    if by_prefix:
        densest = max(
            (e for evs in by_prefix.values() for _, e in evs), key=len, default=None
        )
        if densest and len(densest) > 1:
            dt = densest[-1].wall_time - densest[0].wall_time
            ds = densest[-1].step - densest[0].step
            if ds > 0:
                print(
                    f"\n# throughput: {ds} steps in {dt / 60:.1f} min "
                    f"= {dt / ds:.1f} s/step"
                )


if __name__ == "__main__":
    main()
