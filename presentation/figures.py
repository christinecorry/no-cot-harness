"""Shared plotting helpers: house style, aggregate/CI loaders, and the adaptive-model masking
rule, reused by every figure script in this package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A distinct marker shape per model line, assigned by the model's order in a spec.
MARKERS = ["o", "s", "^", "D", "v", "P"]

_STYLE_PATH = Path(__file__).resolve().parent / "secondlook.mplstyle"


def apply_style() -> None:
    """Apply the house style: the installed name, else the bundled file. The ONE style loader
    every plot entry point calls (this module deliberately doesn't auto-apply on import)."""
    try:
        plt.style.use("secondlook")
    except OSError:
        plt.style.use(str(_STYLE_PATH))


def cond_label(axis: str, value: int) -> str:
    """The stored condition label for a given x-axis value (the anchor maps to baseline)."""
    if axis == "repeat":
        return "baseline" if value == 1 else f"repeat_r{value}"
    return "baseline" if value == 0 else f"filler_f{value}"


def load_accuracy(path: Path) -> Tuple[Dict[Tuple[str, str, str], Optional[float]], dict]:
    data = json.loads(Path(path).read_text())
    acc = {(r["model"], r["dataset"], r["condition"]): r["accuracy"] for r in data["aggregate"]}
    return acc, data["meta"]


def load_cis(path: Path) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    """(model, dataset, condition) -> (ci_lo, ci_hi) for rows that carry a bootstrap CI."""
    data = json.loads(Path(path).read_text())
    return {(r["model"], r["dataset"], r["condition"]): (r["ci_lo"], r["ci_hi"])
            for r in data["aggregate"] if "ci_lo" in r and "ci_hi" in r}


# --- adaptive-model masking --------------------------------------------------------------------
# The adaptive-only models' natural (append) channel refuses/thinks at low augmentation; a
# plotted point there reads as "~0% accuracy" when the truth is "declined to answer". Their
# accuracies are masked to conditions where the model answered immediately (zero reasoning) on at
# least this share of items.
ADAPTIVE_MODELS = {"anthropic/claude-fable-5", "anthropic/claude-opus-5"}
ADAPTIVE_MIN_ANSWER_RATE = 0.85
_adaptive_rates_cache: Dict[tuple, dict] = {}


def adaptive_answer_rates(datasets: tuple) -> dict:
    """(model, dataset, condition) -> share of items that adaptive model answered immediately with
    zero reasoning, over the given dataset ids. Reads the local result store (aggregate counts
    only)."""
    key = tuple(sorted(datasets))
    if key not in _adaptive_rates_cache:
        from harness import sweep
        wanted = set(datasets)
        counts: dict = {}
        with sweep.STORE_PATH.open() as f:
            for line in f:
                if not any(f'"{m}"' in line for m in ADAPTIVE_MODELS):
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a concurrent writer's partial tail line
                if (r.get("model") in ADAPTIVE_MODELS and r.get("dataset") in wanted
                        and r.get("error") is None):
                    # Same signal set as scoring.nocot_violation: tokens AND visible reasoning
                    # content — a provider that returns the content but omits the count must not
                    # read as "answered immediately".
                    u = r.get("usage") or {}
                    answered = (r.get("answer_form") == "immediate"
                                and u.get("reasoning_tokens", 0) == 0
                                and u.get("reasoning_chars", 0) == 0)
                    k = (r["model"], r["dataset"], r["condition"])
                    a, t = counts.get(k, (0, 0))
                    counts[k] = (a + answered, t + 1)
        _adaptive_rates_cache[key] = {k: a / t for k, (a, t) in counts.items() if t}
    return _adaptive_rates_cache[key]


def mask_adaptive(acc: dict, datasets: tuple) -> dict:
    """Drop the adaptive models' low-answer-rate cells from an accuracy dict keyed
    (model, dataset, condition) — masked points render as line gaps, never as fake ~0%."""
    if not any(m in ADAPTIVE_MODELS for (m, _d, _c) in acc):
        return acc
    rates = adaptive_answer_rates(datasets)
    return {k: v for k, v in acc.items()
            if k[0] not in ADAPTIVE_MODELS or rates.get(k, 0.0) >= ADAPTIVE_MIN_ANSWER_RATE}


def acc_pct(acc: dict, model: str, dataset: str, condition: str) -> float:
    """Accuracy as a percentage for plotting; NaN for a missing/masked cell (a line gap)."""
    v = acc.get((model, dataset, condition))
    return v * 100 if v is not None else float("nan")
