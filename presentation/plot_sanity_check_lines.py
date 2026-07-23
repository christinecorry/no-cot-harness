"""Mid-run figures for a condition-matched sweep (`--run`, default `sanity_check_100`), one PNG
per dataset.

Why this reads the raw store instead of the named-run aggregate: `sanity_check_100` was
submitted as THREE SEPARATE `--run sanity_check_100 --models <one-model>` invocations (not one
invocation covering all three models). `run_named` in harness/sweep.py writes
`results/aggregates/sanity_check_100_aggregate.json` keyed only by whichever `--models` subset
was passed *that* invocation, so the file on disk reflects only the last model to finish/write
and does NOT merge the three runs. Until that's fixed (or the sweep completes and the
aggregate step is rerun for all three models together), this script recomputes accuracy
directly from the resumable store (`runs/sweep_store.jsonl`) instead: filter rows to
`error is None`, group by (model, dataset, condition), and only plot a point where n >= 20 (an
in-flight condition can have a handful of landed rows and would be noisy/misleading otherwise).
Rerun this script anytime the sweep advances — it always reflects the current store contents.

Error bars: the repo's usual CI convention (`harness/stats.py`'s `paired_bootstrap_cis`, drawn
via `plot_condition_match.py`'s `load_cis`) is a paired bootstrap over item-level correctness —
that needs the raw per-item hits/misses, which this script's (model, dataset, condition) ->
(correct, n) grouping has already collapsed. So each point here instead gets a 95% Wilson score
interval computed directly from its (correct, n) count: the standard, better-behaved-at-small-n
alternative to a normal approximation, and the defensible simple default for a mid-run sanity
check. Drawn in the same visual style as `plot_condition_match.py`'s CIs (capsize=2, capthick=0.8,
elinewidth=0.8) for consistency with the rest of the repo's figures.

    python -m presentation.plot_sanity_check_lines
    python -m presentation.plot_sanity_check_lines --run condition_matched_500
    python -m presentation.plot_sanity_check_lines --dataset nhop_2
    python -m presentation.plot_sanity_check_lines --min-n 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from harness import config, registry, schema
from presentation.figures import MARKERS, apply_style, cond_label, mask_adaptive

DEFAULT_RUN_NAME = "sanity_check_100"
MIN_N_DEFAULT = 20
WILSON_Z = 1.959964  # 95% two-sided normal quantile

PANELS = [
    ("repeat", "Repeats", "number of problem repeats (r)"),
    ("filler", "Filler", "filler length (f)"),
]


def wilson_ci(correct: int, n: int, z: float = WILSON_Z) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — better-behaved than the normal
    approximation at small n or extreme p, and the right tool when only (correct, n) survive
    (no per-item data to bootstrap over). Returns (lo, hi) as fractions in [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = correct / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_store_counts(store_path: Path, datasets: set) -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    """Group the raw resumable store by (model, dataset, condition), scoring only rows with
    `error is None`. Returns (correct, n) per cell — the full landed counts, unfiltered by
    `min_n` (filtering happens at plot time so coverage reporting still sees everything)."""
    correct: Dict[Tuple[str, str, str], int] = defaultdict(int)
    total: Dict[Tuple[str, str, str], int] = defaultdict(int)
    with store_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a concurrent writer's partial tail line
            if r.get("dataset") not in datasets or r.get("error") is not None:
                continue
            key = (r["model"], r["dataset"], r["condition"])
            total[key] += 1
            if r.get("correct"):
                correct[key] += 1
    return {k: (correct[k], n) for k, n in total.items()}


def describe_coverage(counts: dict, models: List[str], dataset: str, min_n: int) -> str:
    lines = []
    for m in models:
        conds = {c: n for (mm, d, c), (_correct, n) in counts.items() if mm == m and d == dataset}
        if not conds:
            lines.append(f"    {m}: no data yet")
            continue
        ok = sum(1 for n in conds.values() if n >= min_n)
        lines.append(f"    {m}: {ok}/{len(conds)} conditions with n>={min_n} "
                      f"(landed range {min(conds.values())}-{max(conds.values())})")
    return "\n".join(lines)


def plot_one_dataset(dataset: str, axes_spec: dict, counts: dict, min_n: int, models: List[str],
                      out_path: Path, run_name: str, target_n: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (axis_key, title, xlabel) in zip(axes, PANELS):
        values = axes_spec.get(axis_key, [])
        xs = list(range(len(values)))
        for i, m in enumerate(models):
            color = f"C{i}"
            ys, lo_err, hi_err = [], [], []
            for v in values:
                key = (m, dataset, cond_label(axis_key, v) + "+md")
                correct, n = counts.get(key, (0, 0))
                if n < min_n:
                    ys.append(float("nan"))
                    lo_err.append(0.0)
                    hi_err.append(0.0)
                    continue
                acc = correct / n
                ci_lo, ci_hi = wilson_ci(correct, n)
                ys.append(acc * 100)
                lo_err.append((acc - ci_lo) * 100)
                hi_err.append((ci_hi - acc) * 100)
            if all(y != y for y in ys):  # every point NaN -> nothing landed for this model yet
                continue
            ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker=MARKERS[i % len(MARKERS)],
                        markersize=4, linewidth=1.2, linestyle="-", color=color,
                        capsize=2, capthick=0.8, elinewidth=0.8, label=config.short_model(m))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{title} — {dataset}")
        ax.set_ylim(0, 100)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=1, fontsize=8)

    caption = (f"{run_name} (this dataset's target n={target_n}/condition). Points require "
               f"n>={min_n} landed items; error bars are 95% Wilson score intervals on "
               f"(correct, n); gaps mean not enough data has landed yet. Mid-run snapshot, not "
               f"final.")
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8.5, fontweight="semibold", color="#111111")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN_NAME, choices=list(config.NAMED_RUNS),
                    help=f"named run to plot (default: {DEFAULT_RUN_NAME})")
    ap.add_argument("--store", default=str(config.RUNS_DIR / "sweep_store.jsonl"))
    ap.add_argument("--dataset", action="append",
                    help="dataset id to plot (repeatable); default: every dataset in "
                         "the chosen --run's axes")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT,
                    help="minimum landed (non-errored) items required to plot a point")
    ap.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = ap.parse_args(argv)

    apply_style()
    store_path = Path(args.store)
    if not store_path.exists():
        print(f"missing store: {store_path}")
        return 1

    run_axes = config.NAMED_RUNS[args.run]["axes"]
    datasets = args.dataset or list(run_axes.keys())
    models = config.NAMED_RUNS[args.run]["models"]  # fixed order -> stable color/marker per model

    counts = load_store_counts(store_path, set(datasets))
    # mask_adaptive operates on an accuracy dict keyed the same way; build one just for the mask,
    # then drop any (model, dataset, condition) it flags from the raw counts before plotting.
    acc_for_mask = {k: (c / n if n else float("nan")) for k, (c, n) in counts.items()}
    kept_keys = mask_adaptive(acc_for_mask, tuple(datasets)).keys()
    counts = {k: v for k, v in counts.items() if k in kept_keys}

    out_dir = Path(args.out_dir)
    for dataset in datasets:
        axes_spec = run_axes[dataset]
        # NAMED_RUNS[args.run]["n"] can be a take-everything sentinel (e.g. 100000 for
        # condition_matched_500), which isn't the real per-condition target — the dataset's own
        # eval file length is the actual target n for that dataset's panels.
        target_n = min(len(schema.load_jsonl(registry.DATASETS[dataset].eval_path)),
                       config.NAMED_RUNS[args.run]["n"])
        out = out_dir / f"{args.run}_lines_{dataset}.png"
        plot_one_dataset(dataset, axes_spec, counts, args.min_n, models, out, args.run, target_n)
        print(f"wrote {out}")
        print(describe_coverage(counts, models, dataset, args.min_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
