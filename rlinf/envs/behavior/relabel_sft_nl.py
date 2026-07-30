"""Rewrite harvested SFT rows to carry the natural-language goal prompt.

The expert trajectory does not depend on how the goal was *phrased* -- the same
tool sequence satisfies the same BDDL atoms whether the prompt showed the atoms or
an English sentence. So switching the SFT cold-start to the NL-goal condition does
not require re-running OmniGibson over 988 episodes: it only requires re-rendering
``prompt_messages``. That is what this does, sim-free, in a second.

    with-env python -m rlinf.envs.behavior.relabel_sft_nl \\
        --in  /data/behavior-data/sft_mi/behavior_sft_train_mi.jsonl \\
        --out /data/behavior-data/sft_nl/behavior_sft_train_nl.jsonl

Rows whose activity renders with a fallback (see ``nl_goal.render_goal_nl``) are
dropped by default rather than silently trained on -- a mechanical sentence is
worse than no sentence, because it teaches the policy to expect a phrasing no
human would produce. ``--keep-fallbacks`` overrides that.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from rlinf.envs.behavior import sft_build as sb
from rlinf.envs.behavior.nl_goal import render_goal_nl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--bddl-dir", default=None)
    ap.add_argument("--keep-fallbacks", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    cache: dict[str, tuple[str, list[str]]] = {}
    kept = dropped = 0

    with open(args.src) as fin, open(args.dst, "w") as fout:
        for line in fin:
            row = json.loads(line)
            act = row["activity"]
            if act not in cache:
                cache[act] = render_goal_nl(act, bddl_dir=args.bddl_dir)
            goal_nl, fallbacks = cache[act]
            if fallbacks and not args.keep_fallbacks:
                dropped += 1
                continue
            # The goal atoms were rendered into the old prompt; the NL sentence
            # replaces them, and the tool steps are copied through untouched.
            obs_mode = "full"
            for msg in row.get("prompt_messages", []):
                if msg.get("role") == "user":
                    for ln in msg["content"].split("\n"):
                        if ln.startswith("Observability:"):
                            obs_mode = ln.split(":", 1)[1].strip()
            row["prompt_messages"] = sb.build_prompt_messages(
                act, [], obs_mode, goal_nl=goal_nl)
            row["goal_nl"] = goal_nl
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    # The tool-schema sidecar is prompt-independent, so carry it over verbatim
    # rather than making the dataset loader look for it next to the old file.
    side = args.src + ".tools.json"
    if os.path.exists(side):
        shutil.copyfile(side, args.dst + ".tools.json")
    else:
        sb.dump_tool_schemas(args.dst + ".tools.json")

    print(f"activities: {len(cache)}  kept: {kept}  dropped(fallback): {dropped}")
    for act, (text, fb) in sorted(cache.items()):
        mark = "!" if fb else " "
        print(f" {mark} {act}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
