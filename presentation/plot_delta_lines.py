"""Accuracy DELTA (condition minus baseline) line figure — companion to
plot_sanity_check_lines.py's raw-accuracy lines, plotting the lift directly instead of making the
reader subtract baseline by eye. Same two-panel (Repeats | Filler) layout and per-model color/
marker convention, reusing harness/stats.py's paired_ttests for the delta + significance numbers
rather than recomputing anything from the store.

Markers are filled when that point is Holm-corrected significant vs baseline (p<alpha, default
0.001 matching the source LessWrong post's convention), hollow otherwise — so a glance at marker
fill tells you which lift is trustworthy vs noise, the same convention plot_baseline_vs_peak.py
uses via its '*'/unmarked bars.

    python -m presentation.plot_delta_lines
    python -m presentation.plot_delta_lines --dataset nhop_2 --alpha 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from harness import config, stats as hstats
from presentation.figures import MARKERS, apply_style

DEFAULT_RUN_NAME = "condition_matched_500"
PANELS = [("repeat", "Repeats", "number of problem repeats (r)"),
         ("filler", "Filler", "filler length (f)")]


def _cond_value(condition: str) -> int:
    """'repeat_r10+md' -> 10; 'filler_f300+md' -> 300."""
    base = condition.split("+")[0]
    _, _, num = base.partition("_")
    return int(num[1:])


def plot_one_dataset(dataset: str, axes_spec: dict, rows: List[Dict[str, Any]], models: List[str],
                     out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = config.model_colors(models)
    for ax, (axis_key, title, xlabel) in zip(axes, PANELS):
        # Baseline itself (repeat=1, filler=0) has no condition row of its own — delta is
        # trivially 0 there by definition, not a computed statistic — but plotting it anyway as
        # the line's starting point makes the jump from "no augmentation" to the first real
        # condition visible, rather than the line appearing to start already mid-lift.
        anchor = 1 if axis_key == "repeat" else 0
        values = axes_spec.get(axis_key, [])
        if not values:
            ax.axis("off")
            continue
        xs = list(range(len(values)))
        prefix = "repeat_r" if axis_key == "repeat" else "filler_f"
        any_line = False
        for i, m in enumerate(models):
            color = f"#{colors[m]}"
            by_value = {_cond_value(r["condition"]): r for r in rows
                       if r["model"] == m and r["dataset"] == dataset
                       and r["condition"].startswith(prefix)}
            ys, sig = [], []
            for v in values:
                if v == anchor:
                    ys.append(0.0)
                    sig.append(False)  # the anchor is a reference point, not a tested condition
                    continue
                r = by_value.get(v)
                ys.append(r["delta"] * 100 if r else float("nan"))
                sig.append(bool(r["sig_holm"]) if r else False)
            if all(y != y for y in ys):
                continue
            any_line = True
            ax.plot(xs, ys, linewidth=1.2, linestyle="-", color=color,
                   label=config.short_model(m), zorder=2)
            for x, y, s in zip(xs, ys, sig):
                if y != y:
                    continue
                ax.plot(x, y, marker=MARKERS[i % len(MARKERS)], markersize=6,
                       markerfacecolor=color if s else "white", markeredgecolor=color,
                       markeredgewidth=1.2, linestyle="none", zorder=3)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy delta vs baseline (pp)")
        ax.set_title(f"{title} — {config.short_dataset(dataset)}")
        if any_line:
            ax.legend(ncol=1, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN_NAME, choices=list(config.NAMED_RUNS))
    ap.add_argument("--models", help="comma-separated model ids; default: the chosen --run's own list")
    ap.add_argument("--dataset", action="append", help="dataset id (repeatable); default: all in --run")
    ap.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="Holm-corrected significance cutoff for filled vs hollow markers "
                         "(default 0.001, matching plot_baseline_vs_peak.py's convention)")
    args = ap.parse_args(argv)

    apply_style()
    spec: Dict[str, Any] = config.NAMED_RUNS[args.run]
    if args.models:
        spec = {**spec, "models": args.models.split(",")}
    models = spec["models"]
    datasets = args.dataset or list(spec["axes"].keys())

    rows = hstats.paired_ttests(spec, None, alpha=args.alpha)

    out_dir = Path(args.out_dir)
    for dataset in datasets:
        axes_spec = spec["axes"][dataset]
        out = out_dir / f"{args.run}_delta_lines_{dataset}.png"
        plot_one_dataset(dataset, axes_spec, rows, models, out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
