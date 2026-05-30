"""Deterministic language-command definitions for CPU-safe experiment setup.

The first implementation avoids a network dependency on CLIP/SentenceTransformers
while the project is being prepared on CPU. It gives a stable fixed-size vector
per command so observation plumbing can be validated. A frozen text encoder can
replace `embedding_for_text` later without changing the env interface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

LANGUAGE_EMBEDDING_DIM = 16


@dataclass(frozen=True)
class LanguageCommand:
    command_id: int
    text: str
    target: str
    behavior: str


COMMANDS = (
    LanguageCommand(0, "pick up the red cube", "red_cube", "pick"),
    LanguageCommand(1, "pick up the blue cube", "blue_cube", "pick"),
    LanguageCommand(2, "walk to the cube", "cube", "walk"),
    LanguageCommand(3, "stand still", "none", "stand"),
)


def embedding_for_text(text: str, dim: int = LANGUAGE_EMBEDDING_DIM) -> list[float]:
    """Return a deterministic unit-length embedding for a command string."""

    values: list[float] = []
    seed = text.encode("utf-8")
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dim:
                break
        counter += 1
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def command_manifest() -> list[dict[str, object]]:
    """Return command metadata plus deterministic embeddings."""

    rows: list[dict[str, object]] = []
    for command in COMMANDS:
        row = asdict(command)
        row["embedding"] = embedding_for_text(command.text)
        rows.append(row)
    return rows


def write_command_manifest(path: str | Path) -> None:
    """Write JSONL command metadata for dataset generation and logs."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in command_manifest():
            f.write(json.dumps(row) + "\n")

