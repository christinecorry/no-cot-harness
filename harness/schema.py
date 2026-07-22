"""JSONL loading for eval/pool files.

Every dataset record is expected to already be in the canonical shape the rest of the harness
assumes: `{"id", "dataset_id", "problem", "gold_answer", "metadata": {...}}` (see the root
README's "Data" section) — this module just reads it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

Record = Dict[str, Any]


def load_jsonl(path: str | Path) -> List[Record]:
    """Read a JSONL file into a list of records."""
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
