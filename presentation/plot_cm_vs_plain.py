"""Compare condition-matched few-shot demos (this repo) against plain/paper-faithful demos (an
external store — the private replication this repo's condition-matched study extends) for the
same model, one PNG per dataset.

This repo only ever collects condition-matched data (its whole subject), so the plain-demo side
of the comparison has to come from elsewhere. `--plain-store` points at that other store; `--
plain-model` is the model id it uses (may differ from `--model` — e.g. the private store keys the
same model under its bare native id, "claude-opus-4-5", not the OpenRouter alias
"anthropic/claude-opus-4.5" this repo uses). `--dataset-map` lets the two stores disagree on
dataset ids too (this repo's "gen_arithmetic_500"/"comp_math_500" subsets vs the plain store's
full-size "gen_arithmetic"/"comp_math").

The plain store's conditions may carry a temperature suffix ("baseline@t1") from an unrelated
temperature-robustness sweep; only the bare condition labels (no "@" suffix) are the real
plain-demo comparison points, so those are filtered in.

    python -m presentation.plot_cm_vs_plain --plain-store /path/to/other/runs/sweep_store.jsonl
    python -m presentation.plot_cm_vs_plain --plain-store ... --dataset nhop_3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from harness import config, registry, schema
from presentation.figures import apply_style, cond_label
from presentation.plot_sanity_check_lines import wilson_ci

DEFAULT_DATASET_MAP = {
    "gen_arithmetic_500": "gen_arithmetic",
    "comp_math_500": "comp_math",
    "nhop_2": "nhop_2",
    "nhop_3": "nhop_3",
    "nhop_4": "nhop_4",
}

PANELS = [
    ("repeat", "Repeats", "number of problem repeats (r)"),
    ("filler", "Filler", "filler length (f)"),
]


def load_counts(store_path: Path, model: str, datasets: set, *,
                require_md: bool, exclude_temp_suffix: bool
                ) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """(dataset, condition) -> (correct, n), scoped to one model. `require_md`: only "+md"
    conditions (this repo's condition-matched rows) vs only bare conditions (plain demos).
    `exclude_temp_suffix`: drop conditions carrying an unrelated "@t..." temperature-sweep tag."""
    counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    with store_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("model") != model or r.get("dataset") not in datasets or r.get("error") is not None:
                continue
            cond = r["condition"]
            if exclude_temp_suffix and "@" in cond:
                continue
            is_md = cond.endswith("+md")
            if is_md != require_md:
                continue
            key = (r["dataset"], cond[:-3] if is_md else cond)  # strip "+md" for a shared key
            counts[key][1] += 1
            if r.get("correct"):
                counts[key][0] += 1
    return {k: tuple(v) for k, v in counts.items()}


def plot_one_dataset(dataset: str, axes_spec: dict, cm_counts: dict, plain_counts: dict,
                     min_n: int, out_path: Path, model_label: str, target_n: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    series = [("condition-matched (this repo)", cm_counts, "C0", "o"),
              ("plain demos (private replication)", plain_counts, "C1", "s")]
    for ax, (axis_key, title, xlabel) in zip(axes, PANELS):
        values = axes_spec.get(axis_key, [])
        xs = list(range(len(values)))
        for label, counts, color, marker in series:
            ys, lo_err, hi_err = [], [], []
            for v in values:
                key = (dataset, cond_label(axis_key, v))
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
            if all(y != y for y in ys):
                continue
            ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker=marker, markersize=4,
                        linewidth=1.2, linestyle="-", color=color,
                        capsize=2, capthick=0.8, elinewidth=0.8, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{title} — {config.short_dataset(dataset)}")
        ax.set_ylim(0, 100)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=1, fontsize=8)

    caption = (f"{model_label}: condition-matched vs plain demos (this dataset's target "
               f"n={target_n}/condition). Points require n>={min_n} landed items; error bars are "
               f"95% Wilson score intervals on (correct, n).")
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8.5, fontweight="semibold", color="#111111")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="condition_matched_500", choices=list(config.NAMED_RUNS),
                    help="named run this repo's condition-matched side is read from")
    ap.add_argument("--model", default="anthropic/claude-opus-4.5",
                    help="model id as keyed in THIS repo's store")
    ap.add_argument("--plain-model", default="claude-opus-4-5",
                    help="model id as keyed in the plain-demo (--plain-store) store")
    ap.add_argument("--cm-store", default=str(config.RUNS_DIR / "sweep_store.jsonl"))
    ap.add_argument("--plain-store", required=True,
                    help="path to the external store holding plain-demo data")
    ap.add_argument("--dataset", action="append",
                    help="dataset id (this repo's naming, repeatable); default: every dataset in "
                         "the chosen --run's axes")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = ap.parse_args(argv)

    apply_style()
    cm_store = Path(args.cm_store)
    plain_store = Path(args.plain_store)
    for p in (cm_store, plain_store):
        if not p.exists():
            print(f"missing store: {p}")
            return 1

    run_axes = config.NAMED_RUNS[args.run]["axes"]
    datasets = args.dataset or list(run_axes.keys())
    plain_datasets = {DEFAULT_DATASET_MAP.get(d, d) for d in datasets}

    cm_counts_all = load_counts(cm_store, args.model, set(datasets),
                               require_md=True, exclude_temp_suffix=False)
    plain_counts_all = load_counts(plain_store, args.plain_model, plain_datasets,
                                   require_md=False, exclude_temp_suffix=True)

    out_dir = Path(args.out_dir)
    for dataset in datasets:
        plain_dataset = DEFAULT_DATASET_MAP.get(dataset, dataset)
        cm_counts = {k: v for k, v in cm_counts_all.items() if k[0] == dataset}
        # remap the plain side's dataset id to this repo's id so plot_one_dataset can key on one name
        plain_counts = {(dataset, cond): v for (d, cond), v in plain_counts_all.items()
                        if d == plain_dataset}
        target_n = len(schema.load_jsonl(registry.DATASETS[dataset].eval_path))
        out = out_dir / f"cm_vs_plain_{config.short_model(args.model)}_{dataset}.png"
        plot_one_dataset(dataset, run_axes[dataset], cm_counts, plain_counts, args.min_n, out,
                         config.short_model(args.model), target_n)
        print(f"wrote {out}")
        cm_n = sum(n for _, n in cm_counts.values())
        plain_n = sum(n for _, n in plain_counts.values())
        print(f"  cm landed: {cm_n}  plain landed: {plain_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
