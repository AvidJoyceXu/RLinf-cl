"""One debug RGB video and event sidecar per complete BEHAVIOR session."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


class TrajectoryVideoRecorder:
    """Stream timestamp-free tool frames into one session-scoped MP4.

    The event sidecar is the authoritative frame-to-tool alignment. Frames are also
    annotated for quick visual inspection. This recorder is debug evidence only; it
    does not transport images to the policy.
    """

    def __init__(self, output_dir: str, fps: int = 4):
        self.output_dir = os.path.abspath(output_dir)
        self.fps = fps
        self.writer = None
        self.video_path: str | None = None
        self.sidecar_path: str | None = None
        self.metadata: dict = {}
        self.events: list[dict] = []

    def begin(self, metadata: dict) -> str:
        """Open a new recording. Any unfinished prior recording is closed as such."""
        if self.writer is not None:
            self.finish(complete=False, reason="superseded_by_new_session")
        import imageio.v2 as imageio

        os.makedirs(self.output_dir, exist_ok=True)
        stem = "__".join(
            _safe(metadata.get(key))
            for key in ("activity", "instance_id", "session_id")
        )
        self.video_path = os.path.join(self.output_dir, stem + ".mp4")
        self.sidecar_path = os.path.join(self.output_dir, stem + ".json")
        self.metadata = dict(metadata)
        self.events = []
        self.writer = imageio.get_writer(self.video_path, fps=self.fps)
        return self.video_path

    def append(self, frame, *, tool: str, ok: bool | None = None) -> None:
        """Append one rendered frame aligned to a start/tool event."""
        if self.writer is None:
            raise RuntimeError("trajectory recorder has not begun")
        import numpy as np
        from PIL import Image, ImageDraw

        arr = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
        arr = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
        index = len(self.events)
        label = f"frame={index:04d} tool={tool}"
        if ok is not None:
            label += f" ok={bool(ok)}"
        image = Image.fromarray(arr)
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), label)
        draw.rectangle((0, 0, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
        self.writer.append_data(np.asarray(image))
        self.events.append({"frame": index, "tool": tool, "ok": ok})

    def finish(self, *, complete: bool, reason: str = "") -> dict:
        """Close the MP4 and write its event/provenance sidecar."""
        if self.writer is None:
            return {}
        self.writer.close()
        self.writer = None
        record = {
            **self.metadata,
            "complete": bool(complete),
            "reason": reason,
            "fps": self.fps,
            "frames": len(self.events),
            "events": self.events,
            "video_path": self.video_path,
        }
        assert self.sidecar_path is not None
        with open(self.sidecar_path, "w") as stream:
            json.dump(record, stream, indent=2)
        return record


__all__ = ["TrajectoryVideoRecorder"]
