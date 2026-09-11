"""Token-matched dose-response: accuracy points per DOUBLING OF PROMPT LENGTH, with filler and
repeats on the same x-axis (mean prompt tokens actually sent per condition, from the store's
usage fields: input + cache read + cache write on the native transport, input on OpenRouter).
Baseline is included as its own prompt length, so the fit needs no "extra tokens" zero point.
Slope = OLS of per-condition accuracy on log2(prompt tokens); 95% CI by per-item paired
bootstrap (items resampled, per-condition means recomputed, slope refit). Aggregate-level only.

    python3 analysis/token_matched_slopes.py
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harness import config, stats as hstats, sweep

SPEC = config.NAMED_RUNS["blog1_with_astra"]
DATASETS = ["gen_arithmetic_500", "comp_math_500", "nhop_2", "nhop_3", "nhop_4"]
OUT = REPO / "results" / "stats" / "token_matched_slopes.json"
N_BOOT, SEED = 2000, 0

def cond_kind(c):
    b = c.replace("+md", "")
    return "baseline" if b == "baseline" else b.split("_")[0]

def prompt_tokens():
    """(model, dataset, cond-base) -> mean prompt tokens over clean matched rows."""
    tot, cnt = defaultdict(int), defaultdict(int)
    models = set(SPEC["models"])
    with sweep.STORE_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") not in models or r.get("error") is not None: continue
            c = r["condition"]
            if not (c.endswith("+md") or c == "baseline"): continue
            u = r["usage"]
            k = (r["model"], r["dataset"], c.replace("+md", ""))
            tot[k] += u.get("input_tokens", 0) + u.get("cache_read_tokens", 0) + u.get("cache_write_tokens", 0); cnt[k] += 1
    return {k: tot[k] / cnt[k] for k in tot}

def ols(xs, ys):
    xm, ym = xs.mean(), ys.mean(); return float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum())

def main():
    scores = hstats.load_item_scores(SPEC, None)
    toks = prompt_tokens()
    rng = np.random.default_rng(SEED)
    out = []
    for m in SPEC["models"]:
        for d in DATASETS:
            base = scores.get((m, d, "baseline+md")) or scores.get((m, d, "baseline"))
            if not base: continue
            for kind in ("filler", "repeat"):
                ser = [("baseline", base)] + [(c.replace("+md", ""), items) for (mm, dd, c), items in scores.items()
                                              if mm == m and dd == d and cond_kind(c) == kind]
                ser = [(c, it) for c, it in ser if (m, d, c) in toks]
                if len(ser) < 3: continue
                items = sorted(set.intersection(*(set(it) for _, it in ser)))
                M = np.array([[float(it[i]) for i in items] for _, it in ser])
                xs = np.array([math.log2(toks[(m, d, c)]) for c, _ in ser])
                point = ols(xs, M.mean(axis=1))
                boots = [ols(xs, M[:, rng.integers(0, len(items), len(items))].mean(axis=1)) for _ in range(N_BOOT)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                out.append({"model": m, "dataset": d, "axis": kind, "pp_per_doubling_of_prompt_tokens": 100 * point,
                            "ci_lo": 100 * lo, "ci_hi": 100 * hi, "n_items": len(items),
                            "prompt_tokens": {c: round(toks[(m, d, c)]) for c, _ in ser}})
    OUT.write_text(json.dumps({"meta": {"x": "log2(mean prompt tokens per condition, baseline included)", "n_boot": N_BOOT}, "slopes": out}, indent=1) + "\n")
    print(f"{'model':16}{'dataset':20}{'filler pp/dbl':>26}{'repeat pp/dbl':>26}{'ratio r/f':>10}")
    for m in SPEC["models"]:
        for d in DATASETS:
            f_ = next((r for r in out if r["model"] == m and r["dataset"] == d and r["axis"] == "filler"), None)
            r_ = next((r for r in out if r["model"] == m and r["dataset"] == d and r["axis"] == "repeat"), None)
            fmt = lambda s: f"{s['pp_per_doubling_of_prompt_tokens']:+5.2f} [{s['ci_lo']:+.1f},{s['ci_hi']:+.1f}]" if s else "-"
            ratio = (r_["pp_per_doubling_of_prompt_tokens"] / f_["pp_per_doubling_of_prompt_tokens"]) if (f_ and r_ and abs(f_["pp_per_doubling_of_prompt_tokens"]) > 0.5) else float("nan")
            print(f"{config.short_model(m):16}{d:20}{fmt(f_):>26}{fmt(r_):>26}{ratio:>10.1f}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
