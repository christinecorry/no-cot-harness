"""Paired bootstrap confidence intervals and significance tests for sweep accuracy.

Each accuracy point (a given model/dataset/condition) is a mean over the same problem set
used by every other condition in its panel. To put an error bar on it we resample the
*problems* with replacement and watch how much the accuracy moves. The resample draw is
shared across all conditions within a (model, dataset) panel — the paired structure — so the
same "luck of the draw" applies to baseline and augmented conditions alike.

Reads per-item correctness from the local store (runs/sweep_store.jsonl, gitignored). Writes
only numbers (ci_lo / ci_hi, t/p values) back into JSON, so results stay committable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as scistats

from . import config, sweep

# Key for one accuracy point.
Cell = Tuple[str, str, str]  # (model, dataset, condition)


def load_item_scores(spec: Dict[str, Any], method: str | None = None) -> Dict[Cell, Dict[str, bool]]:
    """Per-item correctness for every cell in `spec`, keyed (model, dataset, condition) ->
    {item_id: ok}. `method` is the run's forced CLI method (None = each model's static default).

    Mirrors `sweep.aggregate`: only cells in the spec, and only non-errored rows, are counted.
    """
    wanted = {c["sig"] for c in sweep.enumerate_cells(spec, method)}
    scores: Dict[Cell, Dict[str, bool]] = defaultdict(dict)
    with sweep.STORE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r["sig"] in wanted and r.get("error") is None:
                scores[(r["model"], r["dataset"], r["condition"])][r["item_id"]] = bool(r["correct"])
    return scores


def _split_cond(cond: str) -> Tuple[str, str]:
    """(base condition, match-demos suffix): 'repeat_r10+md' -> ('repeat_r10', '+md'); a
    plain-demo label has an empty suffix. Condition-matched cells are their own panel — a
    condition pairs against the baseline of its OWN demo-rendering (plain vs matched), never
    across the two (a matched condition would otherwise find no literal 'baseline' key and be
    silently skipped, since the matched baseline is stored as 'baseline+md')."""
    base, sep, md = cond.partition("+md")
    return (base, sep + md) if sep else (cond, "")


def _cond_sort_key(cond: str) -> Tuple[int, int, str]:
    """Order conditions: baseline first, then repeats by r, then fillers by f; the match-demos
    suffix orders within each (kind, value) so a panel's rows stay grouped."""
    base, suffix = _split_cond(cond)
    if base == "baseline":
        return (0, 0, suffix)
    kind, _, num = base.partition("_")
    value = int(num[1:]) if num[1:].isdigit() else 0
    return (1 if kind == "repeat" else 2, value, suffix)


def paired_bootstrap_cis(spec: Dict[str, Any], method: str, n_boot: int = 10000, seed: int = 0,
                         alpha: float = 0.05) -> Dict[Cell, Tuple[float, float]]:
    """Return a (lo, hi) accuracy CI for every cell, from a paired bootstrap over problems.

    Within each (model, dataset) we restrict to the problems answered in *all* of that model's
    conditions (the paired set), draw one resample of those problem positions per iteration, and
    apply it to every condition so the conditions move together. CI = the central (1-alpha) band.
    """
    scores = load_item_scores(spec, method)
    rng = np.random.default_rng(seed)
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

    # Group cells by (model, dataset, match-demos suffix); the conditions in a group share one
    # resample draw. Splitting by suffix keeps a plain-demo panel's pairing independent of a
    # condition-matched panel's.
    by_panel: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for (model, dataset, cond) in scores:
        by_panel[(model, dataset, _split_cond(cond)[1])].append(cond)

    cis: Dict[Cell, Tuple[float, float]] = {}
    for (model, dataset, _suffix), conds in by_panel.items():
        # Paired problem set: items present in every condition of this panel.
        common = set.intersection(*(set(scores[(model, dataset, c)]) for c in conds))
        ids = sorted(common)
        if not ids:
            continue
        # rows = conditions, cols = problems; A[c, i] is 1.0 if condition c got problem i right.
        A = np.array([[scores[(model, dataset, c)][i] for i in ids] for c in conds], dtype=np.float64)
        m = A.shape[1]

        boot = np.empty((n_boot, len(conds)), dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, m, m)              # one shared resample of problems...
            boot[b] = A[:, idx].mean(axis=1)         # ...applied to every condition at once
        los, his = np.percentile(boot, [lo_pct, hi_pct], axis=0)
        for c, lo, hi in zip(conds, los, his):
            cis[(model, dataset, c)] = (float(lo), float(hi))
    return cis


def _paired_arrays(a: Dict[str, bool], b: Dict[str, bool]) -> Tuple[np.ndarray, np.ndarray]:
    """Correctness vectors for the problems both conditions answered (paired set)."""
    ids = sorted(set(a) & set(b))
    return (np.array([a[i] for i in ids], dtype=float),
            np.array([b[i] for i in ids], dtype=float))


def _holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_ttests(spec: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    """Paired t-test of each condition vs its baseline, with Holm correction within each panel.

    The family for the Holm adjustment is the set of (condition vs baseline) tests within one
    (model, dataset) panel — i.e. all the r/f points compared to the single baseline of that panel.
    """
    scores = load_item_scores(spec, method)
    by_panel: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for (model, dataset, cond) in scores:
        by_panel[(model, dataset, _split_cond(cond)[1])].append(cond)

    rows: List[Dict[str, Any]] = []
    for (model, dataset, suffix), conds in by_panel.items():
        base_label = "baseline" + suffix
        base = scores.get((model, dataset, base_label))
        if base is None:
            continue
        panel: List[Dict[str, Any]] = []
        for cond in sorted((c for c in conds if c != base_label), key=_cond_sort_key):
            c_arr, b_arr = _paired_arrays(scores[(model, dataset, cond)], base)
            t, p = scistats.ttest_rel(c_arr, b_arr)
            p = 1.0 if np.isnan(p) else float(p)  # zero-variance diffs -> no evidence
            panel.append({
                "model": model, "dataset": dataset, "condition": cond, "n_pairs": len(c_arr),
                "acc_baseline": round(float(b_arr.mean()), 4), "acc_cond": round(float(c_arr.mean()), 4),
                "delta": round(float(c_arr.mean() - b_arr.mean()), 4),
                "t": round(float(t), 3), "p": p,
            })
        for row, p_holm in zip(panel, _holm([r["p"] for r in panel])):
            row["p_holm"] = p_holm
            row["sig_holm"] = p_holm < 0.05
        rows.extend(panel)
    return rows


def write_significance(spec: Dict[str, Any], run_name: str, method: str) -> Path:
    """Write the condition-vs-baseline paired-t table (raw + Holm) to results/ (numbers only).

    Meta records WHEN and over WHAT the table was computed (spec n + per-panel pair counts), so a
    table computed on one item set is detectable against an aggregate later regrown on another —
    without this, a stale significance file silently contradicts its sibling aggregate."""
    from datetime import datetime, timezone
    rows = paired_ttests(spec, method)
    panel_pairs: Dict[str, int] = {}
    for r in rows:
        key = f"{r['model']}/{r['dataset']}"
        panel_pairs[key] = max(panel_pairs.get(key, 0), r["n_pairs"])
    config.STATS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.STATS_DIR / f"{run_name}_significance.json"
    out.write_text(json.dumps({
        "meta": {"test": "paired t-test (scipy ttest_rel), condition vs baseline",
                 "correction": "Holm-Bonferroni within each (model, dataset) panel",
                 "models": spec["models"], "n": spec["n"],
                 "panel_pairs": panel_pairs,
                 "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "significance": rows,
    }, indent=2) + "\n")
    return out


def attach_cis_to_aggregate(aggregate_path: Path, spec: Dict[str, Any], method: str,
                            n_boot: int = 10000, seed: int = 0) -> Path:
    """Add ci_lo / ci_hi / ci_method to each row of an aggregate JSON in place; return the path."""
    path = Path(aggregate_path)
    data = json.loads(path.read_text())
    cis = paired_bootstrap_cis(spec, method, n_boot=n_boot, seed=seed)
    ci_method = f"paired_bootstrap_{n_boot}_pct95"
    for row in data["aggregate"]:
        ci = cis.get((row["model"], row["dataset"], row["condition"]))
        if ci is not None:
            row["ci_lo"], row["ci_hi"] = round(ci[0], 4), round(ci[1], 4)
            row["ci_method"] = ci_method
    data["meta"]["bootstrap"] = {"n_boot": n_boot, "seed": seed, "ci": "percentile_95"}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
