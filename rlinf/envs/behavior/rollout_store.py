"""Crash-safe per-episode persistence for claim-facing rollout jobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def episode_key(**fields: Any) -> str:
    """Return a stable identity for one declared episode."""
    return json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def atomic_write_json(path: str | Path, value: Any, *, indent: int | None = 2) -> None:
    """Atomically replace ``path`` with one fsync'd JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=indent)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class EpisodeStore:
    """Persist episodes independently and materialize JSONL only when complete.

    The sidecar directory is intentionally retained after finalization. It is small,
    records the exact run contract, and makes an interrupted summary/finalization step
    recoverable without rerunning model calls.
    """

    def __init__(
        self,
        output: str | Path,
        *,
        contract: dict[str, Any],
        expected_keys: list[str],
        resume: bool,
    ) -> None:
        self.output = Path(output).resolve()
        self.partial_dir = Path(f"{self.output}.partial")
        self.manifest_path = self.partial_dir / "run.json"
        self.contract = contract
        self.contract_sha256 = canonical_sha256(contract)
        self.expected_keys = list(expected_keys)
        if not self.expected_keys:
            raise ValueError("rollout denominator is empty")
        if len(set(self.expected_keys)) != len(self.expected_keys):
            raise ValueError("rollout denominator contains duplicate episode keys")
        if self.output.exists():
            raise FileExistsError(
                f"refusing to overwrite completed output: {self.output}"
            )

        manifest = {
            "format_version": FORMAT_VERSION,
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
            "expected_keys": self.expected_keys,
        }
        if self.partial_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"partial rollout exists; pass --resume to validate and use it: "
                    f"{self.partial_dir}"
                )
            with self.manifest_path.open(encoding="utf-8") as stream:
                existing = json.load(stream)
            if existing != manifest:
                raise ValueError(
                    "resume contract mismatch: model/data/budget/decoding/code or "
                    "episode denominator changed"
                )
        else:
            self.partial_dir.mkdir(parents=True)
            atomic_write_json(self.manifest_path, manifest)
        self._records = self._load_records()

    @property
    def completed_count(self) -> int:
        return len(self._records)

    def is_complete(self, index: int) -> bool:
        return index in self._records

    def record(self, index: int) -> dict[str, Any]:
        return dict(self._records[index])

    def commit(self, index: int, key: str, record: dict[str, Any]) -> None:
        """Atomically persist one newly completed denominator row."""
        self._validate_index_key(index, key)
        if index in self._records:
            raise FileExistsError(f"episode {index} is already committed")
        stored = {
            **record,
            "episode_index": index,
            "episode_key": key,
            "run_contract_sha256": self.contract_sha256,
        }
        atomic_write_json(self._episode_path(index), stored, indent=None)
        self._records[index] = stored

    def records(self, *, require_complete: bool = False) -> list[dict[str, Any]]:
        if require_complete and len(self._records) != len(self.expected_keys):
            missing = [
                index
                for index in range(len(self.expected_keys))
                if index not in self._records
            ]
            raise ValueError(
                f"rollout denominator incomplete: {len(self._records)}/"
                f"{len(self.expected_keys)}; missing indices {missing[:20]}"
            )
        return [
            dict(self._records[index])
            for index in range(len(self.expected_keys))
            if index in self._records
        ]

    def finalize(self) -> list[dict[str, Any]]:
        """Atomically publish the ordered JSONL after the denominator is complete."""
        records = self.records(require_complete=True)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return records

    def _episode_path(self, index: int) -> Path:
        return self.partial_dir / f"{index:06d}.json"

    def _validate_index_key(self, index: int, key: str) -> None:
        if not 0 <= index < len(self.expected_keys):
            raise IndexError(f"episode index {index} is outside the denominator")
        expected = self.expected_keys[index]
        if key != expected:
            raise ValueError(
                f"episode key mismatch at index {index}: expected {expected!r}, "
                f"got {key!r}"
            )

    def _load_records(self) -> dict[int, dict[str, Any]]:
        records: dict[int, dict[str, Any]] = {}
        for path in sorted(self.partial_dir.glob("*.json")):
            if path == self.manifest_path:
                continue
            if not (path.stem.isdigit() and len(path.stem) == 6):
                raise ValueError(
                    f"unexpected file in rollout partial directory: {path}"
                )
            index = int(path.stem)
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            self._validate_index_key(index, record.get("episode_key"))
            if record.get("episode_index") != index:
                raise ValueError(f"episode index mismatch in {path}")
            if record.get("run_contract_sha256") != self.contract_sha256:
                raise ValueError(f"run contract hash mismatch in {path}")
            records[index] = record
        return records
