"""Run an accuracy sweep synchronously through the OpenRouter API, with a resumable result store.

A sweep evaluates every (model, dataset, method, condition, item) cell. Results are stored in a
JSONL keyed by a signature of that 5-tuple, so a run is idempotent: re-running skips cells that
are already done, and scaling up the item count only evaluates the new cells. Because `method`
(prefill | append | structured) is part of the key, the same model under different methods is
kept as separate, non-colliding result sets — likewise for condition-matched demos (`+md` suffix).

Per-item records (with problem text + raw output) go to the gitignored store under runs/.
Aggregate accuracy (no problem text) is written to results/ and is safe to commit.

Without --method, each (model, dataset) cell uses `registry.resolve_method`'s static default;
--method forces one method everywhere (a controlled comparison).

    python -m harness.sweep --smoke --n 20                          # live e2e check
    python -m harness.sweep --smoke --n 5 --match-demos             # condition-matched e2e check
    python -m harness.sweep --run condition_matched --estimate      # cost table, no submit
    python -m harness.sweep --run condition_matched --max-budget-usd 50
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from . import backends, conditions, config, prompt, registry, schema, scoring

STORE_PATH = config.RUNS_DIR / "sweep_store.jsonl"


def signature(method: str, m: str, dataset: str, cond_label: str, item_id: str) -> str:
    return f"{method}::{m}::{dataset}::{cond_label}::{item_id}"


def _cond_key(cond_label: str, match_demos: bool = False) -> str:
    """The condition label as keyed in the store. Condition-matched-demo cells carry a '+md'
    suffix so they never conflate with plain-demo rows for the same (model, dataset, condition)
    in aggregation."""
    return f"{cond_label}+md" if match_demos else cond_label


def enumerate_cells(spec: Dict[str, Any], forced_method: str | None = None) -> List[Dict[str, Any]]:
    """Expand a sweep spec into individual work cells.

    Each (model, dataset) resolves its own no-CoT method (`registry.resolve_method`; `--method`
    forces one everywhere). A cell carries both the CLI method (`elicitation`, what routes it) and
    the backend's fine-grained store label (`method`, what keys it), so one sweep can mix models
    and methods, with every cell keyed by the channel it actually ran under.
    """
    n = spec["n"]
    cells: List[Dict[str, Any]] = []
    pools: Dict[Path, List[Dict[str, Any]]] = {}  # pool file -> records (per-model pools may repeat)
    for dataset, axes in spec["axes"].items():
        ds = registry.DATASETS[dataset]
        items = schema.load_jsonl(ds.eval_path)[:n]
        conds = config.conditions_for(axes)
        match_demos = bool(axes.get("match_demos"))
        for m in spec["models"]:
            method = registry.resolve_method(m, ds, forced_method)
            label = backends.backend_for(m, method, ds.answer_schema).method
            pool = pools.setdefault(ds.pool_path, schema.load_jsonl(ds.pool_path))
            for cond in conds:
                for item in items:
                    cells.append({
                        "method": label, "elicitation": method, "model": m, "dataset": dataset,
                        "cond": cond, "pool": pool, "item": item, "match_demos": match_demos,
                        "sig": signature(label, m, dataset, _cond_key(cond.label, match_demos),
                                         item["id"]),
                    })
    return cells


def load_done_sigs() -> set:
    """Signatures that completed SUCCESSFULLY (so canceled/errored cells are retried on resume)."""
    if not STORE_PATH.exists():
        return set()
    done = set()
    with STORE_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") is None:
                done.add(r["sig"])
    return done


def _cell_params(cell: Dict[str, Any]) -> Dict[str, Any]:
    return backends.request_params(cell["model"], cell["elicitation"], cell["pool"], cell["item"],
                                   cell["cond"], registry.DATASETS[cell["dataset"]],
                                   match_demos=cell.get("match_demos", False))


def estimate_sync_cost(cells: List[Dict[str, Any]]) -> float:
    """Rough pre-submit estimate (no API call): ~chars/4 input tokens per (model, dataset,
    condition) sample × the count, priced at the model's standard rates."""
    by_combo: Dict[tuple, Dict[str, Any]] = {}
    counts: Dict[tuple, int] = defaultdict(int)
    for c in cells:
        key = (c["model"], c["dataset"], c["cond"].label)
        counts[key] += 1
        by_combo.setdefault(key, c)
    total = 0.0
    for key, sample in by_combo.items():
        mi = registry.model_info(sample["model"])
        out_tokens = registry.DATASETS[sample["dataset"]].max_answer_tokens
        p = _cell_params(sample)
        chars = sum(len(str(msg["content"])) for msg in p["messages"])
        total += counts[key] * (chars / 4 * mi.sync_rate_in + out_tokens * mi.sync_rate_out) / 1e6
    return total


def _score(scorer: Any, parsed: Any, gold: Any, usage: Dict[str, Any],
           tool_violation: Any = None) -> bool:
    """Score a parsed answer with the dataset's scorer; a no-CoT violation (the shared rule in
    `scoring.nocot_violation` — reasoning/thinking tokens, or a structured tool violation) scores
    wrong regardless of the answer produced. Violations are recorded in the row, never turned into
    excluded errors."""
    if scoring.nocot_violation(usage, tool_violation):
        return False
    return scorer.score(parsed, gold)


def _structured_fields(backend: Any, resp: Any) -> Dict[str, Any]:
    """Channel-compliance fields for structured backends (empty dict for the other channels):
    the FULL tool input (so what filled the output budget is visible in the store, not just the
    answer) and the violation verdict."""
    if not hasattr(backend, "tool_violation"):
        return {}
    return {"tool_input": backend.extract_tool_input(resp),
            "tool_violation": backend.tool_violation(resp)}


def _eval_cell(cell: Dict[str, Any], client: Any) -> Dict[str, Any]:
    """Run one cell synchronously: render → assemble prompt → call the model → parse → score.

    Returns a result record (no `sig`/`problem` — the caller adds what it needs). Cell-level
    failures are captured into the record so a batch of cells keeps going. Shared by the resumable
    sync runner and the --smoke check, so both exercise exactly the same evaluation path.
    """
    m, cond, item = cell["model"], cell["cond"], cell["item"]
    ds = registry.DATASETS[cell["dataset"]]
    bk = backends.backend_for(m, cell["elicitation"], ds.answer_schema)
    rec: Dict[str, Any] = {"method": cell["method"], "model": m, "dataset": cell["dataset"],
                           "condition": _cond_key(cond.label, cell.get("match_demos", False)),
                           "item_id": item["id"], "gold": item["gold_answer"]}
    try:
        text = conditions.render(item["problem"], cond)
        messages = prompt.build_messages(cell["pool"], text, bk,
                                         demo_cond=cond if cell.get("match_demos") else None)
        system = prompt.system_for(cond, base=bk.system_base or ds.system_prompt, suffix=ds.filler_suffix)
        resp, out = bk.complete(client, m, messages, system=system, max_tokens=ds.max_answer_tokens)
        parsed = ds.scorer.parse_answer(out)
        usage = bk.usage_dict(resp)
        structured = _structured_fields(bk, resp)
        rec.update(raw_output=out, parsed=parsed,
                   correct=_score(ds.scorer, parsed, item["gold_answer"], usage,
                                  structured.get("tool_violation")),
                   answer_form=ds.scorer.answer_form(out), usage=usage, error=None, **structured)
    except Exception as e:  # noqa: BLE001 - record the failure and keep the run going
        rec.update(raw_output=None, parsed=None, correct=False, answer_form=None, usage={}, error=repr(e))
    return rec


def _clients_for(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One SDK client per distinct backend type across `cells` (reused for every cell of that type).

    `dict.setdefault(key, bk.client())` would evaluate `bk.client()` on EVERY cell regardless of
    whether `key` is already present (Python evaluates a call's arguments before the call) —
    silently constructing and discarding one throwaway SDK/httpx client (and its SSL context) per
    cell instead of once per type. Invisible at pilot scale; at full-run scale (tens of thousands
    of cells) the discarded-client churn alone can burn hours of CPU with zero API calls made.
    Guard the construction explicitly so each type is built exactly once.
    """
    clients: Dict[str, Any] = {}
    for c in cells:
        bk = backends.backend_for(c["model"], c["elicitation"])
        name = type(bk).__name__
        if name not in clients:
            clients[name] = bk.client()
    return clients


def run_sync_cells(cells: List[Dict[str, Any]], *, workers: int = 3,
                   max_calls: int | None = None) -> int:
    """Run cells into the resumable store, concurrently.

    Resumable (skips already-succeeded sigs). Dollar safety lives in the driver's pre-flight
    estimate vs --max-budget-usd; max_calls is a hard ceiling on cells attempted this run.
    """
    done = load_done_sigs()
    todo = [c for c in cells if c["sig"] not in done]
    if max_calls is not None:
        todo = todo[:max_calls]
    if not todo:
        print("  nothing to run (all cells already in store).", flush=True)
        return 0

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    clients = _clients_for(todo)  # one client per backend type, reused across the concurrent run
    store = STORE_PATH.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"written": 0, "errors": 0}

    def _run(cell: Dict[str, Any]) -> None:
        client = clients[type(backends.backend_for(cell["model"], cell["elicitation"])).__name__]
        rec = _eval_cell(cell, client)
        rec["sig"] = cell["sig"]
        with lock:
            store.write(json.dumps(rec, ensure_ascii=False) + "\n")
            store.flush()
            counters["written"] += 1
            counters["errors"] += int(rec["error"] is not None)

    print(f"  running {len(todo)} cells (workers={workers})…", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(_run, c) for c in todo]):
                f.result()  # surface unexpected (uncaught) errors; cell errors are already stored
    finally:
        store.close()
    print(f"  wrote {counters['written']} cells ({counters['errors']} errored).", flush=True)
    return counters["written"]


def tabulate(pairs: Iterable[Tuple[Tuple[str, str, str], bool]], *, ndigits: int = 4
             ) -> List[Dict[str, Any]]:
    """Reduce (key, correct) pairs to sorted accuracy rows, one per (dataset, model, condition)."""
    buckets: Dict[Tuple[str, str, str], List[bool]] = defaultdict(list)
    for key, correct in pairs:
        buckets[key].append(bool(correct))
    rows = [{
        "model": m, "dataset": d, "condition": cond, "n": len(flags),
        "correct": sum(flags),
        "accuracy": round(sum(flags) / len(flags), ndigits) if flags else None,
    } for (d, m, cond), flags in buckets.items()]
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["condition"]))
    return rows


def flag_partial_rows(rows: List[Dict[str, Any]], *, threshold: float = 0.9) -> List[Dict[str, Any]]:
    """Mark rows whose n falls below `threshold` × their (model, dataset) panel's max n.

    A partially-collected condition (e.g. an interrupted run) otherwise pools into the aggregate
    looking like a full data point. Small ragged edges (a few errored-then-excluded items) stay
    unflagged. Returns the flagged rows."""
    panel_max: Dict[Tuple[str, str], int] = {}
    for r in rows:
        key = (r["model"], r["dataset"])
        panel_max[key] = max(panel_max.get(key, 0), r["n"])
    flagged = []
    for r in rows:
        if r["n"] < threshold * panel_max[(r["model"], r["dataset"])]:
            r["partial"] = True
            flagged.append(r)
    for r in flagged:
        print(f"  WARNING partial condition: {r['model']} {r['dataset']} {r['condition']} "
              f"n={r['n']} < {threshold:.0%} of panel max {panel_max[(r['model'], r['dataset'])]}",
              flush=True)
    return flagged


def print_aggregate(rows: List[Dict[str, Any]], *, title: str = "aggregate accuracy") -> None:
    """Print accuracy rows as a compact console table (shared by every run mode)."""
    print(f"\n--- {title} ---")
    print(f"  {'dataset':14} {'model':22} {'condition':14} {'n':>5} {'correct':>7} {'acc':>7}")
    for r in rows:
        print(f"  {r['dataset']:14} {config.short_model(r['model']):22} {r['condition']:14} "
              f"{r['n']:>5} {r['correct']:>7} {str(r['accuracy']):>7}")


def _model_methods(cells: List[Dict[str, Any]]) -> Dict[str, str]:
    """Per-model store label(s) actually used across `cells` (joined if a model spans labels)."""
    by_model: Dict[str, set] = defaultdict(set)
    for c in cells:
        by_model[c["model"]].add(c["method"])
    return {m: ",".join(sorted(labels)) for m, labels in sorted(by_model.items())}


def aggregate(spec: Dict[str, Any], run_name: str, forced_method: str | None = None
              ) -> List[Dict[str, Any]]:
    """Aggregate accuracy by (model, dataset, condition) for this spec; write to results/."""
    cells = enumerate_cells(spec, forced_method)
    wanted = {c["sig"] for c in cells}
    # Dedup by signature (last clean row wins) — a duplicate clean sig (e.g. two concurrent
    # resumes) must not double-count in the accuracy denominator.
    best: Dict[str, dict] = {}
    with STORE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            # Errored cells (e.g. API rejections) are missing data points, not wrong answers —
            # exclude them so accuracy isn't deflated by failures unrelated to the model.
            if r["sig"] in wanted and r.get("error") is None:
                best[r["sig"]] = r
    pairs = [((r["dataset"], r["model"], r["condition"]), bool(r["correct"]))
             for r in best.values()]
    rows = tabulate(pairs, ndigits=4)
    flag_partial_rows(rows)

    model_methods = _model_methods(cells)
    labels = sorted(set(model_methods.values()))
    config.AGGREGATES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.AGGREGATES_DIR / f"{run_name}_aggregate.json"
    meta = {"method": labels[0] if len(labels) == 1 else ",".join(labels),
            "model_methods": model_methods, "models": spec["models"], "n": spec["n"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps({"meta": meta, "aggregate": rows}, indent=2) + "\n")
    return rows


def estimate_table(spec: Dict[str, Any], forced_method: str | None = None) -> float:
    """Print a per-(model, dataset) cost estimate for a spec and return the grand total.

    Input-dominated: counts ~chars/4 input tokens for one sample per (model, dataset, condition)
    × the cell count × the model's input rate. Filler conditions carry the big token counts, so
    this is a realistic upper-ish bound. Output (~10-40 tokens/call) is negligible and omitted.
    """
    cells = enumerate_cells(spec, forced_method)
    groups: Dict[tuple, Dict[str, Any]] = {}
    for c in cells:
        g = groups.setdefault((c["model"], c["dataset"], c["cond"].label), {"count": 0, "sample": c})
        g["count"] += 1
    tok: Dict[tuple, int] = {}
    for key, g in groups.items():
        p = _cell_params(g["sample"])
        tok[key] = sum(len(str(msg["content"])) for msg in p["messages"]) // 4

    print("\n=== cost estimate (input tokens × cells × rate; no cache, output omitted) ===")
    print(f"  {'model':22} {'dataset':14} {'conds':>5} {'items':>6} {'cells':>7} {'avg tok/call':>12} "
          f"{'rate $/M':>9} {'est $':>8}")
    grand = 0.0
    for m in spec["models"]:
        m_cost = 0.0
        mi = registry.model_info(m)
        rate = mi.sync_rate_in
        for d in spec["axes"]:
            keys = [k for k in groups if k[0] == m and k[1] == d]
            if not keys:
                continue
            cells_n = sum(groups[k]["count"] for k in keys)
            items = cells_n // len(keys)
            cost = sum(tok[k] * groups[k]["count"] * rate / 1e6 for k in keys)
            avg = sum(tok[k] * groups[k]["count"] for k in keys) / cells_n
            m_cost += cost
            print(f"  {config.short_model(m):22} {d:14} {len(keys):>5} {items:>6} {cells_n:>7} "
                  f"{avg:>12.0f} {rate:>9.2f} {cost:>8.2f}")
        print(f"  {'':22} {'subtotal ' + config.short_model(m):14}{'':27} {'':>9} {m_cost:>8.2f}")
        grand += m_cost
    print(f"  {'':22} {'GRAND TOTAL':14}{'':36} {'':>9} {grand:>8.2f}")
    print(f"  (+35% contingency -> ~${grand * 1.35:.2f})")
    return grand


# --- Smoke check (the live e2e gate) -----------------------------------------------------------
# A few items per (model, dataset) over one baseline, one repeat, one filler. Runs SYNCHRONOUSLY
# and does NOT write the resumable store — a throwaway correctness check, not a measurement run.
SMOKE_AXES = {"repeat": [1, 5], "filler": [0, 100]}


def _acc_pairs(records: List[Dict[str, Any]]) -> List[Tuple[tuple, bool]]:
    return [((r["dataset"], r["model"], r["condition"]), bool(r["correct"])) for r in records]


def _print_smoke_report(records: List[Dict[str, Any]], agg: List[Dict[str, Any]],
                        methods: Dict[str, str], n: int, show: int) -> None:
    errors = [r for r in records if r["error"]]
    print("=== Smoke (live, synchronous; resumable store NOT written) ===")
    print(f"  methods={methods}  n/dataset={n}  calls={len(records)}  errors={len(errors)}")

    print(f"\n--- per-item (first {show}) ---")
    print(f"  {'dataset':14} {'model':16} {'condition':14} {'gold':>12} {'parsed':>12}  ok  problem")
    for r in records[:show]:
        prob = r["problem"]
        prob = prob if len(prob) <= 56 else prob[:53] + "…"
        ok = "✓" if r["correct"] else "✗"
        print(f"  {r['dataset']:14} {config.short_model(r['model']):16} {r['condition']:14} "
              f"{str(r['gold']):>12} {str(r['parsed']):>12}  {ok}   {prob!r}")

    print_aggregate(agg, title="aggregate accuracy (dataset × model × condition)")

    # No-CoT confirmation from token usage. reasoning_chars covers providers that return visible
    # reasoning CONTENT while omitting the token count (the same signal scoring.nocot_violation
    # penalizes) — a token-only check would pass them as compliant.
    usages = [r["usage"] for r in records if r["usage"]]
    reasoning = sum(u.get("reasoning_tokens", 0) for u in usages)
    reasoning_chars = sum(u.get("reasoning_chars", 0) for u in usages)
    inp = sum(u.get("input_tokens", 0) for u in usages)
    print(f"\n--- no-CoT --- reasoning_tokens={reasoning} reasoning_chars={reasoning_chars} "
          f"(both must be 0 — proves no chain-of-thought)")
    print(f"--- usage --- uncached_input={inp}")

    print("--- no-CoT compliance BY MODEL (answer_form; want immediate==all, reasoning_first==0) ---")
    for m in sorted({r["model"] for r in records}):
        mrecs = [r for r in records if r["model"] == m]
        forms: Dict[str, int] = defaultdict(int)
        for r in mrecs:
            forms[r["answer_form"] or "error"] += 1
        rtoks = sum((r["usage"] or {}).get("reasoning_tokens", 0) for r in mrecs)
        rchars = sum((r["usage"] or {}).get("reasoning_chars", 0) for r in mrecs)
        immediate = forms.get("immediate", 0)
        verdict = ("COMPLIANT" if immediate == len(mrecs) and rtoks == 0 and rchars == 0
                   else "CHECK — reasons/empty")
        print(f"  {config.short_model(m):22} {dict(forms)}  reasoning_tokens={rtoks}  "
              f"reasoning_chars={rchars}  immediate={immediate}/{len(mrecs)}  -> {verdict}")

    # WHAT KIND of non-compliance, not just whether: refusal (channel/prompt problem) and hidden
    # reasoning (measurement problem — reasoning_tokens>0 with no visible trace) need different
    # follow-up, and `answer_form`/`reasoning_tokens` alone don't distinguish them at a glance.
    print("--- response kind BY MODEL (scoring.classify_response) ---")
    for m in sorted({r["model"] for r in records}):
        kinds: Dict[str, int] = defaultdict(int)
        for r in (r for r in records if r["model"] == m):
            kinds[scoring.classify_response(r)] += 1
        print(f"  {config.short_model(m):22} {dict(kinds)}")

    if errors:
        print("\n--- errors (first 3) ---")
        for r in errors[:3]:
            print(f"  {r['dataset']}/{config.short_model(r['model'])}/{r['condition']} "
                  f"{r['item_id']}: {r['error']}")


def run_smoke(n: int, models: List[str], datasets: List[str], show: int,
              forced_method: str | None = None, match_demos: bool = False) -> int:
    """Live synchronous correctness check (the e2e gate): exercise the full eval path — prompt
    assembly, no-CoT enforcement, parsing, scoring — on a small slice, print a per-item table +
    aggregate, and confirm reasoning_tokens==0. Full per-item records (with problem text) go to the
    gitignored store under runs/; an aggregate-only summary (no problem text) goes to results/.
    `match_demos`: condition-matches every dataset's smoke axes (see `prompt.build_messages`)."""
    axes = dict(SMOKE_AXES, match_demos=True) if match_demos else dict(SMOKE_AXES)
    spec = {"models": models, "n": n, "axes": {d: dict(axes) for d in datasets}}
    cells = enumerate_cells(spec, forced_method)
    clients = _clients_for(cells)
    records = []
    for cell in cells:
        rec = _eval_cell(cell, clients[type(backends.backend_for(cell["model"], cell["elicitation"])).__name__])
        rec["problem"] = cell["item"]["problem"]  # for the per-item table + full (gitignored) save
        records.append(rec)

    methods = _model_methods(cells)
    agg = tabulate(_acc_pairs(records), ndigits=3)
    _print_smoke_report(records, agg, methods, n, show)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.AGGREGATES_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = config.RUNS_DIR / f"pilot_{ts.replace(':', '').replace('-', '')}.jsonl"
    with runs_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"meta": {"generated_at_utc": ts, "n_per_dataset": n, "models": models,
                        "datasets": datasets, "methods": methods,
                        "conditions": [c.label for c in config.conditions_for(SMOKE_AXES)]},
               "aggregate": agg}
    summary_path = config.AGGREGATES_DIR / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsaved: {runs_path} (full, gitignored)  |  {summary_path} (aggregate)")
    return 0 if not any(r["error"] for r in records) else 2


# --- Named-run driver (unattended end-to-end) -------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def run_named(args: argparse.Namespace) -> int:
    """Drive a named run end-to-end (run remaining cells → aggregate → report)."""
    run_name = args.run
    spec = config.NAMED_RUNS[run_name]
    if args.n is not None:
        spec = {**spec, "n": args.n}  # local override; don't mutate the module-level spec
    _log(f"run={run_name} n={spec['n']} models={[config.short_model(m) for m in spec['models']]}")

    cells = enumerate_cells(spec, args.method)
    done = load_done_sigs()  # read the store ONCE, not per cell
    todo = [c for c in cells if c["sig"] not in done]
    _log(f"cells={len(cells)} done={len(cells) - len(todo)} todo={len(todo)}")

    if todo:
        if args.max_budget_usd is not None:
            est = estimate_sync_cost(todo)
            _log(f"estimated cost ≈ ${est:.2f} (cap ${args.max_budget_usd:.2f})")
            if est > args.max_budget_usd:
                _log("ABORT: estimate exceeds --max-budget-usd")
                return 3
        _log(f"running {len(todo)} cells (workers={args.workers})…")
        _log(f"collected {run_sync_cells(todo, workers=args.workers, max_calls=args.max_calls)} results")
    else:
        _log("nothing to run; aggregating from store.")

    rows = aggregate(spec, run_name, args.method)
    print_aggregate(rows)
    if args.no_report:
        _log("DONE (--no-report: skipped significance + CIs)")
        return 0

    # Analysis outputs only (committable, no presentation dependency): the significance table and
    # bootstrap CIs written back into the aggregate. Figures are rendered separately
    # (presentation/figures.py, presentation/plot_condition_match.py).
    from . import stats  # local import: stats imports sweep, so avoid a top-level cycle
    agg_path = config.AGGREGATES_DIR / f"{run_name}_aggregate.json"
    sig_path = stats.write_significance(spec, run_name, args.method)
    stats.attach_cis_to_aggregate(agg_path, spec, args.method)
    _log(f"wrote {sig_path.name} + bootstrap CIs (render figures via presentation/)")
    _log("DONE")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Accuracy sweep: live smoke check or a named run")
    ap.add_argument("--smoke", action="store_true",
                    help="live synchronous correctness check (the e2e gate); store NOT written")
    ap.add_argument("--run", choices=list(config.NAMED_RUNS),
                    help="run a named sweep end-to-end (run remaining cells, aggregate, report)")
    ap.add_argument("--estimate", action="store_true",
                    help="with --run: print the per-(model, dataset) cost table and exit (no submit)")
    ap.add_argument("--method", default=None, choices=["prefill", "append", "structured"],
                    help="force ONE method everywhere (a controlled comparison); "
                         "default: each model's static default in registry.py")
    ap.add_argument("--max-budget-usd", type=float, default=None, help="abort if estimate exceeds this")
    ap.add_argument("--workers", type=int, default=3, help="concurrent workers (default 3)")
    ap.add_argument("--max-calls", type=int, default=None, help="hard ceiling on cells this run")
    ap.add_argument("--no-report", action="store_true", help="skip significance + CIs (just aggregate)")
    ap.add_argument("--n", type=int, default=None, help="items/dataset (--smoke default 8; overrides a run's n)")
    ap.add_argument("--models", help="comma-separated model ids for --smoke")
    ap.add_argument("--datasets", help="comma-separated dataset ids for --smoke")
    ap.add_argument("--show", type=int, default=18, help="per-item rows --smoke prints (default 18)")
    ap.add_argument("--match-demos", action="store_true",
                    help="condition-match every few-shot demo to the query's repeat/filler "
                         "condition (opt-in; default renders demos bare)")
    args = ap.parse_args(argv)

    if args.smoke:
        models = args.models.split(",") if args.models else config.MODELS
        datasets = args.datasets.split(",") if args.datasets else config.SMOKE_DEFAULT_DATASETS
        return run_smoke(args.n or 8, models, datasets, args.show, args.method, args.match_demos)

    if args.run:
        if args.estimate:
            spec = config.NAMED_RUNS[args.run]
            if args.n is not None:
                spec = {**spec, "n": args.n}
            estimate_table(spec, args.method)
            return 0
        return run_named(args)

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
