"""Submit/collect cells through the native Anthropic Message Batches API (50% off both input and
output vs sync — see the root README's transport note) instead of a live call per cell.

Batch is inherently async (create -> poll -> fetch results, "up to 24 hours" per Anthropic's own
docs), so it doesn't fit `sweep.py`'s synchronous `run_named` loop. This module reuses the exact
same cell construction and request params (`backends.request_params(..., transport=
"anthropic_native")`) so a batch-collected row is built from an IDENTICAL prompt to its sync
counterpart — only the transport differs — and writes into the SAME resumable store
(`runs/sweep_store.jsonl`) with the same row shape `sweep._eval_cell` produces, so batch and sync
rows merge into one aggregate without any special-casing downstream.

    python -m harness.anthropic_batch --submit --run sanity_check_100 --models anthropic/claude-opus-4.5
    python -m harness.anthropic_batch --poll <batch_id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from . import backends, config, registry
from .sweep import STORE_PATH, _cond_key, _score, _structured_fields, enumerate_cells


def _client() -> Any:
    import anthropic
    return anthropic.Anthropic()


# A cell's `method` label encodes which TRANSPORT collected it (e.g. "openrouter_prefill" vs
# "anthropic_native_prefill" — see backends.py), but a batch-collected row should NOT re-collect
# an (model, dataset, condition, item) already gathered via OpenRouter for the same underlying
# elicitation: prompts were verified byte-identical between transports for both models' actual
# methods (prefill / tool), so a plain sig match (which is transport-specific) would silently
# re-pay for everything OpenRouter already has. Normalize to the logical method family for dedup.
_METHOD_FAMILY = {
    "openrouter_prefill": "prefill", "anthropic_native_prefill": "prefill",
    "openrouter_tool": "structured", "anthropic_native_tool": "structured",
    "openrouter_adaptive": "append", "append_noreason": "append", "anthropic_native_append": "append",
    "structured_json": "structured",
}


def _content_key(model: str, dataset: str, condition: str, item_id: str, method: str) -> tuple:
    return (model, dataset, condition, item_id, _METHOD_FAMILY.get(method, method))


def _covered_content_keys() -> set:
    """(model, dataset, condition, item_id, method_family) already collected SUCCESSFULLY by
    ANY transport — the cross-transport dedup set for batch submission (see `_METHOD_FAMILY`)."""
    covered = set()
    if not STORE_PATH.exists():
        return covered
    with STORE_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") is None:
                covered.add(_content_key(r["model"], r["dataset"], r["condition"], r["item_id"],
                                         r["method"]))
    return covered


def _short_id(sig: str) -> str:
    """The Batches API caps `custom_id` at 64 chars; our signatures run longer. A SHA-256 hex
    digest is exactly 64 chars and deterministic from `sig`, so both submit and collect can
    derive it independently without persisting a separate mapping."""
    return hashlib.sha256(sig.encode()).hexdigest()


def _batch_request(cell: Dict[str, Any]) -> Dict[str, Any]:
    """One Batches API request entry: `custom_id` is a hash of the cell's own store signature
    (see `_short_id`), so a batch result maps straight back into the resumable store's usual
    keying."""
    ds = registry.DATASETS[cell["dataset"]]
    params = backends.request_params(cell["model"], cell["elicitation"], cell["pool"], cell["item"],
                                     cell["cond"], ds, match_demos=cell.get("match_demos", False),
                                     transport="anthropic_native")
    return {"custom_id": _short_id(cell["sig"]), "params": params}


# A Message Batch caps at 100,000 requests OR 256 MB, whichever comes first (Anthropic's own
# batch-processing docs). This repo's condition-matched cells can run well past 20KB/request at
# high filler/repeat, so at full 500-item scale the REQUEST COUNT stays far under 100k while the
# byte size is what actually binds — chunk by measured size, not by a fixed count, with real
# margin below the hard cap since request sizes vary a lot cell to cell.
_MAX_BATCH_BYTES = 200_000_000


def _chunk_cells_by_size(cells: List[Dict[str, Any]],
                         max_bytes: int = _MAX_BATCH_BYTES) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = 0
    for c in cells:
        size = len(json.dumps(_batch_request(c)).encode())
        if current and current_bytes + size > max_bytes:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(c)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def submit_batch(cells: List[Dict[str, Any]]) -> str:
    """Submit one Anthropic Message Batch covering `cells` (every cell must be an anthropic/* id
    — native transport only; must already fit under the API's 100k-request/256MB cap — see
    `_chunk_cells_by_size` for splitting a larger cell list). Returns the batch id; nothing is
    written to the store yet."""
    non_anthropic = [c["model"] for c in cells if not c["model"].startswith("anthropic/")]
    if non_anthropic:
        raise ValueError(f"anthropic_batch only supports anthropic/* ids, got: {set(non_anthropic)}")
    client = _client()
    batch = client.messages.batches.create(requests=[_batch_request(c) for c in cells])
    return batch.id


def submit_batches(cells: List[Dict[str, Any]]) -> List[str]:
    """Split `cells` into API-limit-sized chunks (see `_chunk_cells_by_size`) and submit one
    Message Batch per chunk. Returns all batch ids, in submission order."""
    return [submit_batch(chunk) for chunk in _chunk_cells_by_size(cells)]


def poll_and_collect(batch_id: str, cells_by_sig: Dict[str, Dict[str, Any]], *,
                     poll_interval_s: float = 30.0) -> int:
    """Block until `batch_id` finishes, then map every result into the same row shape
    `sweep._eval_cell` produces and append to the resumable store. Returns the count written."""
    client = _client()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        print(f"  batch {batch_id}: {batch.processing_status} "
              f"({batch.request_counts.processing} processing, "
              f"{batch.request_counts.succeeded} succeeded) — waiting…", flush=True)
        time.sleep(poll_interval_s)

    cells_by_short_id = {_short_id(sig): cell for sig, cell in cells_by_sig.items()}

    written = 0
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("a", encoding="utf-8") as store:
        for entry in client.messages.batches.results(batch_id):
            cell = cells_by_short_id[entry.custom_id]
            ds = registry.DATASETS[cell["dataset"]]
            bk = backends.backend_for(cell["model"], cell["elicitation"], ds.answer_schema,
                                      transport="anthropic_native")
            item = cell["item"]
            rec: Dict[str, Any] = {"method": cell["method"], "model": cell["model"],
                                   "dataset": cell["dataset"], "condition": _cond_key(
                                       cell["cond"].label, cell.get("match_demos", False)),
                                   "item_id": item["id"], "gold": item["gold_answer"],
                                   "sig": cell["sig"]}
            if entry.result.type != "succeeded":
                rec.update(raw_output=None, parsed=None, correct=False, answer_form=None,
                          usage={}, error=f"batch result type: {entry.result.type}")
            else:
                resp = entry.result.message
                out = bk.extract_text(resp)
                parsed = ds.scorer.parse_answer(out)
                usage = bk.usage_dict(resp)
                structured = _structured_fields(bk, resp)
                rec.update(raw_output=out, parsed=parsed,
                          correct=_score(ds.scorer, parsed, item["gold_answer"], usage,
                                         structured.get("tool_violation")),
                          answer_form=ds.scorer.answer_form(out), usage=usage, error=None,
                          **structured)
            store.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit/collect a native Anthropic Batch")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--poll", metavar="BATCH_ID(S)", help="one batch id, or comma-separated several")
    ap.add_argument("--run", choices=list(config.NAMED_RUNS))
    ap.add_argument("--models", help="comma-separated anthropic/* model ids")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args(argv)

    if args.submit:
        spec = config.NAMED_RUNS[args.run]
        if args.models:
            spec = {**spec, "models": args.models.split(",")}
        if args.n is not None:
            spec = {**spec, "n": args.n}
        cells = enumerate_cells(spec, None, transport="anthropic_native")
        covered = _covered_content_keys()
        todo = [c for c in cells
               if _content_key(c["model"], c["dataset"],
                                _cond_key(c["cond"].label, c.get("match_demos", False)),
                                c["item"]["id"], c["method"]) not in covered]
        print(f"submitting {len(todo)} cells ({len(cells) - len(todo)} already covered by ANY "
              f"transport)…")
        batch_ids = submit_batches(todo)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{ts}] {len(batch_ids)} batch(es): {','.join(batch_ids)}")
        print(f"  poll with --poll {','.join(batch_ids)}")
        return 0

    if args.poll:
        # Re-derive the same cell set to map custom_id -> cell at collection time.
        spec = config.NAMED_RUNS[args.run] if args.run else None
        if spec is None:
            raise SystemExit("--poll needs --run (+ optionally --models/--n) to rebuild the cell map")
        if args.models:
            spec = {**spec, "models": args.models.split(",")}
        if args.n is not None:
            spec = {**spec, "n": args.n}
        cells = enumerate_cells(spec, None, transport="anthropic_native")
        cells_by_sig = {c["sig"]: c for c in cells}
        total_written = 0
        for batch_id in args.poll.split(","):
            total_written += poll_and_collect(batch_id, cells_by_sig)
        print(f"wrote {total_written} rows to {STORE_PATH}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
