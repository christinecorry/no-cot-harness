"""Re-score every stored response and assert the result matches what the store recorded.

This is the $0 regression gate for the harness: after any change to parsing/scoring plumbing,
replaying the sweep store must reproduce the stored `parsed`, `answer_form`, and `correct`
fields exactly — proving the change didn't alter scoring semantics on a single one of the
already-paid-for responses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.scoring import rescore_record as rescore  # noqa: E402  the ONE shared re-scoring rule


def replay(store_path: Path) -> int:
    """Replay one store; return the number of mismatching records (0 = parity)."""
    checked = skipped = 0
    mismatches: List[str] = []
    with store_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            rec = json.loads(line)
            if rec.get("error") is not None:
                skipped += 1
                continue
            checked += 1
            got = rescore(rec)
            diffs = {k: (rec.get(k), v) for k, v in got.items() if rec.get(k) != v}
            if diffs:
                mismatches.append(f"  line {line_no} [{rec['sig']}]: " +
                                  "; ".join(f"{k} stored={s!r} replayed={r!r}" for k, (s, r) in diffs.items()))

    print(f"{store_path}: {checked} replayed, {skipped} skipped (errored), {len(mismatches)} mismatches")
    for m in mismatches[:20]:
        print(m)
    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches) - 20} more")
    return len(mismatches)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", nargs="*", type=Path,
                        default=[REPO_ROOT / "runs/sweep_store.jsonl"],
                        help="store files to replay (default: the sweep store)")
    args = parser.parse_args(argv)

    total_mismatches = 0
    for p in args.stores:
        if not p.exists():
            raise SystemExit(f"missing store: {p}")
        total_mismatches += replay(p)
    print("REPLAY PASS" if total_mismatches == 0 else "REPLAY FAIL")
    return 0 if total_mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
