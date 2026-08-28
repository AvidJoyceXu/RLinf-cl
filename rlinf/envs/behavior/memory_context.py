"""Public-only context utilities for the TextWorld memory benchmark."""

from __future__ import annotations

import json
from typing import Any


def extract_public_memory(tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract only scene/landmark/category facts visible in tool payloads.

    View-local handles, activity-local tracks, boxes, scores, scope names, and metric
    positions are intentionally discarded. The result is safe to place in the summary
    arm because every retained field was already present in policy observations.
    """
    scene_id = None
    landmarks: dict[str, dict[str, Any]] = {}
    for turn in tool_trace or []:
        payload = turn.get("payload") or turn.get("result") or {}
        if not isinstance(payload, dict):
            continue
        view = payload.get("view") or {}
        observed_scene = view.get("scene_id")
        if observed_scene:
            if scene_id is not None and scene_id != observed_scene:
                raise ValueError(
                    f"one memory trace crossed scenes: {scene_id} != {observed_scene}"
                )
            scene_id = str(observed_scene)
        view_landmark = view.get("landmark") or {}
        landmark_id = view_landmark.get("id")
        if landmark_id:
            row = landmarks.setdefault(
                str(landmark_id), {"categories": set(), "rooms": set()}
            )
            if view_landmark.get("category"):
                row["categories"].add(str(view_landmark["category"]))
            if view_landmark.get("room"):
                row["rooms"].add(str(view_landmark["room"]))
        for detection in payload.get("detections") or []:
            landmark_id = detection.get("landmark")
            category = detection.get("category")
            if not landmark_id or not category:
                continue
            row = landmarks.setdefault(
                str(landmark_id), {"categories": set(), "rooms": set()}
            )
            row["categories"].add(str(category))
    return {
        "scene_id": scene_id,
        "landmarks": {
            landmark: {
                "categories": sorted(value["categories"]),
                "rooms": sorted(value["rooms"]),
            }
            for landmark, value in sorted(landmarks.items(), key=_landmark_order)
        },
    }


def _landmark_order(item: tuple[str, Any]) -> tuple[int, str]:
    label = item[0]
    return (
        int(label[1:]) if label.startswith("l") and label[1:].isdigit() else 10**9,
        label,
    )


def merge_public_memory(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge two public summaries, refusing cross-scene contamination."""
    left_scene = left.get("scene_id")
    right_scene = right.get("scene_id")
    if left_scene and right_scene and left_scene != right_scene:
        raise ValueError(
            f"cannot merge different scenes: {left_scene} != {right_scene}"
        )
    scene_id = left_scene or right_scene
    combined: dict[str, dict[str, set[str]]] = {}
    for memory in (left, right):
        for landmark, row in (memory.get("landmarks") or {}).items():
            target = combined.setdefault(
                str(landmark), {"categories": set(), "rooms": set()}
            )
            target["categories"].update(str(x) for x in row.get("categories") or [])
            target["rooms"].update(str(x) for x in row.get("rooms") or [])
    return {
        "scene_id": scene_id,
        "landmarks": {
            landmark: {
                "categories": sorted(row["categories"]),
                "rooms": sorted(row["rooms"]),
            }
            for landmark, row in sorted(combined.items(), key=_landmark_order)
        },
    }


def render_public_memory(memory: dict[str, Any]) -> str:
    """Render a deterministic compact summary for the ``summary`` arm."""
    lines = [f"Scene memory for {memory.get('scene_id') or 'unknown-scene'}:"]
    for landmark, row in (memory.get("landmarks") or {}).items():
        rooms = ", ".join(row.get("rooms") or []) or "room unknown"
        categories = ", ".join(row.get("categories") or []) or "nothing recorded"
        lines.append(f"- {landmark} ({rooms}): {categories}")
    if len(lines) == 1:
        lines.append("- no landmark associations recorded")
    return "\n".join(lines)


def summary_messages(memory: dict[str, Any]) -> list[dict[str, str]]:
    """A complete neutral exchange that can precede the next activity prompt."""
    return [
        {
            "role": "user",
            "content": "Recall only the public scene memory from earlier activities.",
        },
        {"role": "assistant", "content": render_public_memory(memory)},
    ]


def flatten_context_blocks(blocks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [message for block in blocks for message in block]


def rendered_context_tokens(tokenizer, messages: list[dict], tools: list[dict]) -> int:
    """Count the exact request prefix under the rollout chat template."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def compact_context_blocks(
    tokenizer,
    *,
    system_message: dict,
    current_user_message: dict,
    blocks: list[list[dict[str, Any]]],
    tools: list[dict],
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop oldest whole activity blocks until the target request fits."""
    for first_kept in range(len(blocks) + 1):
        prior = flatten_context_blocks(blocks[first_kept:])
        messages = [system_message, *prior, current_user_message]
        tokens = rendered_context_tokens(tokenizer, messages, tools)
        if tokens <= max_tokens:
            return prior, {
                "prefix_tokens": tokens,
                "blocks_total": len(blocks),
                "blocks_kept": len(blocks) - first_kept,
                "blocks_dropped": first_kept,
            }
    raise ValueError(
        "current activity prompt exceeds memory context cap: "
        + json.dumps({"max_tokens": max_tokens}, sort_keys=True)
    )


__all__ = [
    "compact_context_blocks",
    "extract_public_memory",
    "flatten_context_blocks",
    "merge_public_memory",
    "render_public_memory",
    "rendered_context_tokens",
    "summary_messages",
]
