"""Dose-response panels behind the post's "Utilization of filler tokens / problem repeats" picker.

One PNG per (dataset, show, view) under results/figures_with_astra/prompt_length/:
  dataset: the all-datasets mean, comp, gen, 2/3/4-hop
  show:    both   -> x = mean prompt tokens (log), filler solid/filled, repeats dashed/hollow
           filler -> x = filler count f (levels evenly spaced)
           repeat -> x = repeat count r (levels evenly spaced)
  view:    acc    -> accuracy with Wilson 95% intervals
           delta  -> change vs. baseline (pp) with paired-bootstrap 95% intervals

Accuracy and deltas come from the sweep store (per-item scores via `stats.load_item_scores`);
mean prompt tokens per condition from results/stats/token_matched_slopes.json (written by
`token_matched_slopes.py`, which reads the provider-reported prompt_tokens of every row). The
"all datasets" panels average, per model and condition, over the conditions present in every
dataset with equal weight per dataset; their intervals combine the per-dataset half-widths in
quadrature divided by the number of datasets.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness import config, stats as hstats  # noqa: E402
from presentation.figures import MARKERS, apply_style  # noqa: E402

SPEC = config.NAMED_RUNS["blog1_with_astra"]
MODELS = SPEC["models"]
DATASETS = ["comp_math_500", "gen_arithmetic_500", "nhop_2", "nhop_3", "nhop_4"]
TOKENS = REPO / "results" / "stats" / "token_matched_slopes.json"
OUT = REPO / "results" / "figures_with_astra" / "prompt_length"
SHOW = {"both": ("filler", "repeat"), "filler": ("filler",), "repeat": ("repeat",)}
STYLE = {"filler": dict(ls="-", hollow=False), "repeat": dict(ls="--", hollow=True)}
N_BOOT, SEED = 2000, 0

# One point: (cond, tokens, acc, lo, hi, delta, dlo, dhi), all in percentage points.


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (c - h), 100 * (c + h)


def load() -> dict:
    """dataset -> model -> axis ('filler'|'repeat') -> [point] sorted by mean prompt tokens."""
    tok = {(r["model"], r["dataset"], r["axis"]): r["prompt_tokens"]
           for r in json.load(open(TOKENS))["slopes"]}
    scores = hstats.load_item_scores(SPEC, None)
    rng = np.random.default_rng(SEED)
    pts: dict = {}
    for (m, d, k), cond_tokens in tok.items():
        if m not in MODELS or d not in DATASETS:
            continue
        base = scores.get((m, d, "baseline+md")) or scores.get((m, d, "baseline"))
        rows = []
        for c, t in cond_tokens.items():
            items = scores.get((m, d, c + "+md")) or scores.get((m, d, c))
            if not items or not base:
                continue
            a, lo, hi = wilson(sum(items.values()), len(items))
            if c == "baseline":
                rows.append((c, t, a, lo, hi, 0.0, 0.0, 0.0))
                continue
            ids = [i for i in items if i in base]
            diff = np.array([float(items[i]) - float(base[i]) for i in ids])
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
            dlo, dhi = np.percentile(boots, [2.5, 97.5])
            rows.append((c, t, a, lo, hi, 100 * diff.mean(), 100 * dlo, 100 * dhi))
        pts.setdefault(d, {}).setdefault(m, {})[k] = sorted(rows, key=lambda q: q[1])
    return pts


def _mean(qs, i):
    return sum(q[i] for q in qs) / len(qs)


def _quad(qs, i, j):
    """Half-widths combined in quadrature, divided by the number of datasets."""
    return math.sqrt(sum((q[i] - q[j]) ** 2 for q in qs)) / len(qs)


def series(pts, m, d, k):
    if d != "__all__":
        return pts.get(d, {}).get(m, {}).get(k)
    byc: dict[str, list] = {}
    for dd in DATASETS:
        for q in pts.get(dd, {}).get(m, {}).get(k, []):
            byc.setdefault(q[0], []).append(q)
    out = []
    for c, qs in byc.items():
        if len(qs) < len(DATASETS):
            continue
        a, dl = _mean(qs, 2), _mean(qs, 5)
        out.append((c, _mean(qs, 1), a, a - _quad(qs, 2, 3), a + _quad(qs, 4, 2),
                    dl, dl - _quad(qs, 5, 6), dl + _quad(qs, 7, 5)))
    return sorted(out, key=lambda q: q[1]) or None


def dose(cond, k):
    if cond == "baseline":
        return 0 if k == "filler" else 1
    return int(cond.split("_")[1][1:])


def title_for(d, show, view):
    ds = "all five datasets (mean)" if d == "__all__" else config.short_dataset(d)
    y = "no-CoT accuracy" if view == "acc" else "change vs. baseline"
    x = {"both": "prompt length", "filler": "filler count", "repeat": "repeat count"}[show]
    return f"{ds}: {y} by {x}"


def plot(pts, d, show, view, out):
    colors = config.model_colors(MODELS)
    ks = SHOW[show]
    levels = sorted({dose(q[0], k) for k in ks for m in MODELS for q in series(pts, m, d, k) or []})
    fig, ax = plt.subplots(figsize=(9, 5.4))
    vi = (2, 3, 4) if view == "acc" else (5, 6, 7)
    for i, m in enumerate(MODELS):
        for k in ks:
            s = series(pts, m, d, k)
            if not s:
                continue
            c, st = f"#{colors[m]}", STYLE[k]
            xs = [q[1] for q in s] if show == "both" else [levels.index(dose(q[0], k)) for q in s]
            ax.errorbar(
                xs, [q[vi[0]] for q in s],
                yerr=[[q[vi[0]] - q[vi[1]] for q in s], [q[vi[2]] - q[vi[0]] for q in s]],
                color=c, linestyle=st["ls"], linewidth=2.0 if m.endswith("astra") else 1.4,
                marker=MARKERS[i % len(MARKERS)], markersize=5,
                markerfacecolor="white" if st["hollow"] else c, markeredgecolor=c, markeredgewidth=1.2,
                capsize=2.5, capthick=0.8, elinewidth=0.8,
            )
    if show == "both":
        ax.set_xscale("log")
        ax.set_xlabel("mean prompt tokens (log scale)")
    else:
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(v) for v in levels])
        ax.set_xlabel("filler tokens (f)" if show == "filler" else "problem repeats (r)")
        ax.margins(x=0.05)
    if view == "acc":
        ax.set_ylim(0, 100)
        ax.set_ylabel("accuracy (%)")
    else:
        ax.axhline(0, color="#999", lw=0.8, zorder=0)
        ax.set_ylabel("change vs. baseline (pp)")
    ax.set_title(title_for(d, show, view), fontsize=12, fontweight="bold")
    h = [Line2D([], [], color=f"#{colors[m]}", marker=MARKERS[i % len(MARKERS)], lw=1.6, label=config.short_model(m))
         for i, m in enumerate(MODELS)]
    if show == "both":
        h += [Line2D([], [], color="#444", ls="-", marker="o", label="filler tokens"),
              Line2D([], [], color="#444", ls="--", marker="o", markerfacecolor="white", label="problem repeats")]
    ax.legend(handles=h, loc="best", ncol=2, fontsize=10, frameon=True, framealpha=0.9, edgecolor="#dddddd",
              handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    pts = load()
    for d in ["__all__", *DATASETS]:
        for show in SHOW:
            for view in ("acc", "delta"):
                plot(pts, d, show, view, OUT / f"{'all' if d == '__all__' else d}__{show}__{view}.png")
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
