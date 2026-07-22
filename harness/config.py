"""Shared configuration: paths, models, condition grids, and the named-run registry."""
from __future__ import annotations

from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent
DATA_DIR = REPO_ROOT / "data"          # eval/pool JSONL files — gitignored, not distributed (see README)
RUNS_DIR = REPO_ROOT / "runs"          # full per-item outputs (problem text) — gitignored
RESULTS_DIR = REPO_ROOT / "results"    # aggregate metrics only — committable, no problem text
FIGURES_DIR = RESULTS_DIR / "figures"
AGGREGATES_DIR = RESULTS_DIR / "aggregates"  # per-run accuracy aggregates (+ bootstrap CIs)
STATS_DIR = RESULTS_DIR / "stats"            # significance tables (paired t-tests)

# The three models this repo studies, all via the OpenRouter alias namespace (see the root
# README's transport note for why).
MODELS = ["anthropic/claude-opus-4.5", "openai/gpt-5.6-sol", "anthropic/claude-fable-5"]

# Condition grids, per dataset family: repeat counts (1 == baseline) and count-to-N filler
# lengths (0 == baseline). Every run spec below draws its axes from these dicts.
GEN_AXES = {"repeat": [1, 2, 3, 5, 10, 20, 40], "filler": [0, 30, 100, 300, 1000]}
COMP_AXES = {"repeat": [1, 2, 3, 5, 10], "filler": [0, 30, 100, 300, 1000, 3000]}
HEADLINE_AXES = {"repeat": [1, 2, 3, 5, 10, 20, 40], "filler": [0, 10, 30, 100, 300, 1000]}

# Dataset rows (paths, scorer, prompts, answer schema) live in `harness/registry.py` — one row per
# dataset. How each model is made to answer immediately (no chain-of-thought) is owned by its
# backend in `harness/backends.py`; which method a (model, dataset) pair uses by default is
# `registry.resolve_method`'s job (`--method` overrides it for any run).

SMOKE_DEFAULT_DATASETS = ["gen_arithmetic", "comp_math"]


def short_model(model: str) -> str:
    """A compact label for tables, e.g. 'anthropic/claude-opus-4.5' -> 'opus-4.5'."""
    return model.split("/")[-1].replace("claude-", "")


def conditions_for(axes: dict):
    """Distinct conditions implied a dataset's repeat/filler axes (baseline counted once)."""
    from . import conditions as C
    conds = [C.baseline()]
    conds += [C.repeat(r) for r in axes.get("repeat", []) if r > 1]
    conds += [C.filler(f) for f in axes.get("filler", []) if f > 0]
    return conds


# --- Named runs -------------------------------------------------------------------------------
# One dict per named sweep: `models`, `n` (items per dataset), and `axes` (dataset id -> condition
# grid, optionally with `"match_demos": True` to condition-match every few-shot demo to the
# query's repeat/filler condition — see `prompt.build_messages`). `--run <name> --estimate` prints
# a cost table before anything is submitted.

# The plain-demo baseline (paper-faithful default: only the query is augmented) — a control to
# compare condition-matched results against on the same items/conditions.
PLAIN_DEMOS = {
    "models": MODELS,
    "n": 100000,
    "axes": {
        "gen_arithmetic": dict(GEN_AXES),
        "comp_math": dict(COMP_AXES),
        "nhop_2": dict(HEADLINE_AXES),
        "nhop_3": dict(HEADLINE_AXES),
        "nhop_4": dict(HEADLINE_AXES),
    },
}

# The condition-matched variant of the same grid — this repo's subject. Condition-matching
# multiplies prompt length for every augmented condition (every demo now carries the query's
# repeat/filler load too, not just the query) — run `--estimate` before submitting.
CONDITION_MATCHED = {
    "models": MODELS,
    "n": 100000,
    "axes": {
        "gen_arithmetic": dict(GEN_AXES, match_demos=True),
        "comp_math": dict(COMP_AXES, match_demos=True),
        "nhop_2": dict(HEADLINE_AXES, match_demos=True),
        "nhop_3": dict(HEADLINE_AXES, match_demos=True),
        "nhop_4": dict(HEADLINE_AXES, match_demos=True),
    },
}

# Same as CONDITION_MATCHED, but gen_arithmetic uses its fixed 1,000-item subset
# (`gen_arithmetic_1000` in registry.py) instead of the full 2,990 — gen_arithmetic is the
# single biggest cost line item and, being generated data with no natural scarcity, is the one
# dataset worth trimming; comp_math and n-hop stay at their full, fixed-size native counts.
CONDITION_MATCHED_TRIMMED = {
    "models": MODELS,
    "n": 100000,
    "axes": {
        "gen_arithmetic_1000": dict(GEN_AXES, match_demos=True),
        "comp_math": dict(COMP_AXES, match_demos=True),
        "nhop_2": dict(HEADLINE_AXES, match_demos=True),
        "nhop_3": dict(HEADLINE_AXES, match_demos=True),
        "nhop_4": dict(HEADLINE_AXES, match_demos=True),
    },
}

# A tiny slice for `sweep --pilot`-style smoke testing without spending much.
PILOT_SPEC = {
    "models": MODELS,
    "n": 5,
    "axes": {"gen_arithmetic": {"repeat": [1, 5], "filler": [0, 100]}},
}

# Sanity check before committing to full scale: n=100/dataset, condition-matched, using the
# 500-item subsets for gen_arithmetic/comp_math (cheaper than their full/1000-item counts for a
# check that isn't the final measurement) and the full n-hop sets (no smaller tier exists there).
SANITY_CHECK_100 = {
    "models": MODELS,
    "n": 100,
    "axes": {
        "gen_arithmetic_500": dict(GEN_AXES, match_demos=True),
        "comp_math_500": dict(COMP_AXES, match_demos=True),
        "nhop_2": dict(HEADLINE_AXES, match_demos=True),
        "nhop_3": dict(HEADLINE_AXES, match_demos=True),
        "nhop_4": dict(HEADLINE_AXES, match_demos=True),
    },
}

NAMED_RUNS: dict[str, dict] = {
    "pilot": PILOT_SPEC,
    "plain_demos": PLAIN_DEMOS,
    "condition_matched": CONDITION_MATCHED,
    "condition_matched_trimmed": CONDITION_MATCHED_TRIMMED,
    "sanity_check_100": SANITY_CHECK_100,
}
