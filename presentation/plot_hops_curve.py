"""Accuracy vs number of hops — one figure, every model as a pair of lines: DASHED with hollow
markers for the no-CoT baseline, SOLID with filled markers for the model's best repeat-or-filler
condition on that hop count (whichever peak scored higher, chosen independently per hop count,
same rule as plot_figure1.py's `_best_peak`). Reuses plot_baseline_vs_peak.py's compute_peaks
(paired t-tests over the store) rather than recomputing anything; error bars are Wilson 95%
intervals on each point's own n. Same per-model color/marker convention as every other figure
in this package, so a model looks the same here as in its bar and delta figures.

    python -m presentation.plot_hops_curve --run blog1_with_astra --out results/figures_with_astra/hops_baseline_vs_best.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from harness import config
from presentation.figures import MARKERS, apply_style
from presentation.plot_baseline_vs_peak import DEFAULT_RUN_NAME, Panel, _cond_short_label, compute_peaks
from presentation.plot_figure1 import _best_peak


def _hops(dataset: str) -> Optional[int]:
    return int(dataset.split("_")[1]) if dataset.startswith("nhop_") else None


def _wilson(acc: float, n: Optional[int], z: float = 1.96) -> Tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return (acc - (centre - half), (centre + half) - acc)


def plot_hops_curve(peaks: Dict[Tuple[str, str], Panel], models: List[str], datasets: List[str],
                    out_path: Path, *, best_only: bool = False,
                    baseline_for: Optional[List[str]] = None, direct_labels: bool = False,
                    title: Optional[str] = None) -> Path:
    """`best_only` drops every baseline line; `baseline_for` re-adds the dashed baseline for
    just those models (a highlight, e.g. astra alone); `direct_labels` names each model at
    its right-hand end instead of relying on the legend alone (readable up to ~4 models)."""
    baseline_models = set(models) if not best_only else set(baseline_for or [])
    hop_datasets = sorted((d for d in datasets if _hops(d)), key=_hops)
    xs = [_hops(d) for d in hop_datasets]
    colors = config.model_colors(models)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    chosen: Dict[str, List[str]] = {}
    end_labels: List[Tuple[float, str]] = []   # (y at the last hop, name) for direct labels
    for i, model in enumerate(models):
        color, marker = f"#{colors[model]}", MARKERS[i % len(MARKERS)]
        base_y, base_err, best_y, best_err, labels = [], [[], []], [], [[], []], []
        for d in hop_datasets:
            baseline_acc, pr, pf = peaks.get((model, d), (None, None, None))
            if baseline_acc is None:
                base_y.append(float("nan")); best_y.append(float("nan")); labels.append("-")
                for e in (base_err, best_err):
                    e[0].append(0.0); e[1].append(0.0)
                continue
            peak = _best_peak(baseline_acc, pr, pf)
            n = (peak or pr or pf or {}).get("n_pairs")
            base_y.append(baseline_acc * 100)
            lo, hi = _wilson(baseline_acc, n); base_err[0].append(lo * 100); base_err[1].append(hi * 100)
            acc = peak["acc_cond"] if peak else baseline_acc
            best_y.append(acc * 100)
            lo, hi = _wilson(acc, n); best_err[0].append(lo * 100); best_err[1].append(hi * 100)
            labels.append(_cond_short_label(peak["condition"]) if peak else "-")
        chosen[model] = labels
        if model in baseline_models:
            ax.errorbar(xs, base_y, yerr=base_err, color=color, linestyle="--", linewidth=1.4,
                        marker=marker, markersize=6, markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.4, capsize=3, capthick=1.0, elinewidth=0.8, alpha=0.9,
                        zorder=3)
        # capthick is set explicitly: matplotlib defaults it to markeredgewidth, which the house
        # style zeroes, so caps silently vanished on the filled-marker series.
        ax.errorbar(xs, best_y, yerr=best_err, color=color, linestyle="-", linewidth=2.0,
                    marker=marker, markersize=6, capsize=3, capthick=1.0, elinewidth=0.8, zorder=4)
        if direct_labels and best_y and best_y[-1] == best_y[-1]:
            end_labels.append((best_y[-1], config.short_model(model)))

    # Direct labels at the right end, spread apart so two models that finish within a few
    # points of each other (sol and fable both ~4% at 4-hop) don't overprint.
    min_gap = 4.5
    placed = []
    for y, name in sorted(end_labels):
        y_lab = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(y_lab)
        ax.annotate(name, (xs[-1], y), xytext=(10, (y_lab - y) * 2.2), textcoords="offset points",
                    va="center", fontsize=9, color="#222222",
                    arrowprops=None if y_lab == y else dict(arrowstyle="-", color="#999999",
                                                            lw=0.6, shrinkA=0, shrinkB=2))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h}-hop" for h in xs])
    ax.set_xlim(xs[0] - 0.25, xs[-1] + (0.6 if direct_labels else 0.25))
    ax.set_ylim(0, 100)
    ax.set_xlabel("number of hops")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title or ("No-CoT accuracy vs hops: best augmentation per model" if best_only
                           else "No-CoT accuracy vs hops: baseline and best augmentation"))

    model_handles = [Line2D([], [], color=f"#{colors[m]}", marker=MARKERS[i % len(MARKERS)],
                            linestyle="-", label=config.short_model(m))
                     for i, m in enumerate(models)]
    style_handles = [Line2D([], [], color="#555555", linestyle="-", marker="o",
                            label="best repeat/filler condition"),
                     Line2D([], [], color="#555555", linestyle="--", marker="o",
                            markerfacecolor="white", markeredgewidth=1.4, label="baseline (no augmentation)")]
    if not baseline_models:
        style_handles = style_handles[:1]
    leg1 = ax.legend(handles=model_handles, loc="upper right", fontsize=8, title="model",
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=8)   # clear of both the model legend and the 2-hop points

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    for m, labels in chosen.items():
        print(f"  {config.short_model(m):14} best condition at " +
              ", ".join(f"{h}-hop={lab}" for h, lab in zip(xs, labels)))
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN_NAME, choices=list(config.NAMED_RUNS))
    ap.add_argument("--models", help="comma-separated model ids; default: the chosen --run's own list")
    ap.add_argument("--out", default=str(config.FIGURES_DIR / "hops_baseline_vs_best.png"))
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--best-only", action="store_true", help="drop the dashed baseline lines")
    ap.add_argument("--baseline-for", help="comma-separated model ids that keep their baseline "
                                           "line under --best-only (a highlight)")
    ap.add_argument("--direct-labels", action="store_true", help="name each model at its right end")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    apply_style()
    spec = config.NAMED_RUNS[args.run]
    models = args.models.split(",") if args.models else list(spec["models"])
    datasets = [d for d in spec["axes"] if d.startswith("nhop_")]
    spec = {**spec, "models": models, "axes": {d: spec["axes"][d] for d in datasets}}
    peaks = compute_peaks(spec, alpha=args.alpha)
    out = plot_hops_curve(peaks, models, datasets, Path(args.out), best_only=args.best_only,
                          baseline_for=args.baseline_for.split(",") if args.baseline_for else None,
                          direct_labels=args.direct_labels, title=args.title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
