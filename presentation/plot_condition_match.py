"""Plain-demo vs condition-matched accuracy, per model, one figure per dataset.

Reads the two named runs' aggregates (`plain_demos`, `condition_matched` — see
`harness/config.py`'s `NAMED_RUNS`) and draws each model twice: solid for the paper-faithful
plain-demo design, dashed for the condition-matched variant (every few-shot demo rendered
through the query's repeat/filler condition too — see `harness/prompt.py`'s `build_messages`).
Both lines are drawn on the SAME two-panel layout (repeats | filler) so the comparison is direct.

    python -m presentation.plot_condition_match
    python -m presentation.plot_condition_match --dataset comp_math
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from harness import config
from presentation.figures import (
    MARKERS, acc_pct, apply_style, cond_label, load_accuracy, load_cis, mask_adaptive,
)

REPEAT_VALUES = [1, 2, 3, 5, 10, 20, 40]
FILLER_VALUES = [0, 10, 30, 100, 300, 1000]


def plot_one_dataset(dataset: str, plain_path: Path, matched_path: Path, out_path: Path) -> Path:
    plain_acc, plain_meta = load_accuracy(plain_path)
    plain_cis = load_cis(plain_path)
    matched_acc, matched_meta = load_accuracy(matched_path)
    matched_cis = load_cis(matched_path)

    plain_acc = mask_adaptive(plain_acc, (dataset,))
    matched_acc = mask_adaptive(matched_acc, (dataset,))

    models = plain_meta["models"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [
        ("repeat", REPEAT_VALUES, "Repeats", "number of problem repeats (r)"),
        ("filler", FILLER_VALUES, "Filler", "filler length (f)"),
    ]

    for ax, (axis, values, title, xlabel) in zip(axes, panels):
        xs = list(range(len(values)))
        for i, m in enumerate(models):
            color = f"C{i}"
            for acc, cis, style, label_suffix in (
                (plain_acc, plain_cis, "-", "plain demos"),
                (matched_acc, matched_cis, "--", "matched demos"),
            ):
                ys, lo_err, hi_err = [], [], []
                for v in values:
                    key = (m, dataset, cond_label(axis, v) + ("+md" if label_suffix == "matched demos" else ""))
                    y = acc_pct(acc, *key)
                    ys.append(y)
                    ci = cis.get(key)
                    a = acc.get(key)
                    if ci is not None and a is not None:
                        lo_err.append((a - ci[0]) * 100)
                        hi_err.append((ci[1] - a) * 100)
                    else:
                        lo_err.append(0.0)
                        hi_err.append(0.0)
                label = f"{config.short_model(m)} ({label_suffix})"
                if any(e for e in lo_err + hi_err):
                    ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker=MARKERS[i % len(MARKERS)],
                               markersize=4, linewidth=1.2, linestyle=style, color=color,
                               capsize=2, capthick=0.8, elinewidth=0.8, label=label)
                else:
                    ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], markersize=4,
                           linewidth=1.2, linestyle=style, color=color, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{title} — {dataset}")
        ax.set_ylim(0, 100)
        ax.legend(ncol=1, fontsize=7)

    caption = "Solid: plain (unaugmented) demos.  Dashed: every demo condition-matched to the query."
    fig.text(0.5, 0.005, caption, ha="center", fontsize=9, fontweight="semibold", color="#111111")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plain-aggregate", default=str(config.AGGREGATES_DIR / "plain_demos_aggregate.json"))
    ap.add_argument("--matched-aggregate", default=str(config.AGGREGATES_DIR / "condition_matched_aggregate.json"))
    ap.add_argument("--dataset", action="append",
                    help="dataset id to plot (repeatable); default: every dataset in the plain-demo aggregate")
    ap.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = ap.parse_args(argv)

    apply_style()
    plain_path, matched_path = Path(args.plain_aggregate), Path(args.matched_aggregate)
    for p in (plain_path, matched_path):
        if not p.exists():
            print(f"missing aggregate: {p} — run the named sweep first (see the root README).")
            return 1

    datasets = args.dataset
    if not datasets:
        acc, _ = load_accuracy(plain_path)
        datasets = sorted({d for (_m, d, _c) in acc})

    out_dir = Path(args.out_dir)
    for dataset in datasets:
        out = out_dir / f"figure_condition_match_{dataset}.png"
        plot_one_dataset(dataset, plain_path, matched_path, out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
