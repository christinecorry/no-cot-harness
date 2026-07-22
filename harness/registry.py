"""Registries: which datasets and models exist, and how a (model × dataset) pair is elicited no-CoT.

Two orthogonal axes, so adding a dataset or a model is one row here:
  - `Dataset` bundles everything dataset-specific the driver needs (paths, scorer, prompts,
    structured answer schema, token cap) — the driver itself stays dataset-agnostic.
  - `Model` consolidates provider + pricing.
  - `resolve_method` picks the no-CoT method per (model, dataset): the cheapest clean channel,
    filtered by capability, overridden where a channel is empirically non-compliant.

A dataset's `scorer` is anything exposing `parse_answer / score / answer_form`; every answer type
lives in `harness.scoring` as a small class, and the driver duck-types on the three methods.

This repo ships three models via the OpenRouter alias namespace — an adaptive-thinking-only model
(cannot have its internal reasoning pass disabled via API parameter) and two models with the
standard prefill/append/structured no-CoT channels — to study condition-matched few-shot demos.
See the root README for why.

Datasets are NOT included in this repo (no generator, no data — see the README's "Data" section);
`eval_path`/`pool_path` just describe where the harness expects to find them locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import config, prompt, scoring

# n-hop dataset paths — not included in this repo (see the README's "Data" section); the harness
# expects these files to exist locally if you supply your own n-hop data.
_NHOP_DATA_DIR = config.DATA_DIR / "nhop"
_NHOP_HOPS = (2, 3, 4)
_NHOP_EVAL = {h: _NHOP_DATA_DIR / f"nhop_{h}_eval.jsonl" for h in _NHOP_HOPS}
_NHOP_POOL = _NHOP_DATA_DIR / "nhop_pool10.jsonl"


@dataclass(frozen=True)
class Dataset:
    id: str                          # the store-signature dataset field — must never change
    eval_path: Path
    pool_path: Path
    scorer: Any                      # anything exposing parse_answer / score / answer_form
                                     # (a harness.scoring scorer instance)
    system_prompt: str               # base instruction; adaptive backends override via system_base
    filler_suffix: str               # filler explanation ("problem" vs "question" wording)
    answer_schema: str = "integer"   # "integer" | "string" — picks the structured tool/json literal
    strict_prompt: Optional[str] = None  # adaptive-channel strict instruction adapted to this
                                         # dataset's answer type; None = the backend's math wording
    max_answer_tokens: int = 50


@dataclass(frozen=True)
class Model:
    id: str
    provider: str                    # "anthropic" | "openai" | "openrouter"
    batch_input_rate: Optional[float]  # $/M batch input; None = provider has no batch endpoint
    sync_rate_in: float              # $/M standard input
    sync_rate_out: float             # $/M standard output


DATASETS: Dict[str, Dataset] = {
    "gen_arithmetic": Dataset(
        id="gen_arithmetic",
        eval_path=config.DATA_DIR / "gen_arithmetic" / "eval_full.jsonl",
        pool_path=config.DATA_DIR / "gen_arithmetic" / "fewshot_pool.jsonl",
        scorer=scoring.INTEGER,
        system_prompt=prompt.SYSTEM_PROMPT,
        filler_suffix=prompt.FILLER_INSTRUCTION_SUFFIX,
    ),
    # A FIXED 1,000-item subset of gen_arithmetic — a separate saved file (not a `--n` runtime
    # slice), so the same 1,000 items run every time regardless of what `--n` is passed elsewhere.
    # Random sample, seed 42, not the first 1,000: gold-answer magnitudes are heavily tailed (a
    # few problems produce huge results), and a contiguous slice measurably under-represents that
    # tail (empirically checked: first-1000 mean |gold| ~1.7M vs the full set's ~7.2M; this random
    # sample's mean ~7.0M tracks the full set closely) even though generation order itself has no
    # difficulty structure (each of the 3000 problems is an independent draw from one seeded RNG).
    "gen_arithmetic_1000": Dataset(
        id="gen_arithmetic_1000",
        eval_path=config.DATA_DIR / "gen_arithmetic" / "eval_1000.jsonl",
        pool_path=config.DATA_DIR / "gen_arithmetic" / "fewshot_pool.jsonl",
        scorer=scoring.INTEGER,
        system_prompt=prompt.SYSTEM_PROMPT,
        filler_suffix=prompt.FILLER_INSTRUCTION_SUFFIX,
    ),
    "comp_math": Dataset(
        id="comp_math",
        eval_path=config.DATA_DIR / "comp_math" / "eval.jsonl",
        pool_path=config.DATA_DIR / "comp_math" / "fewshot_pool.jsonl",
        scorer=scoring.INTEGER,
        system_prompt=prompt.SYSTEM_PROMPT,
        filler_suffix=prompt.FILLER_INSTRUCTION_SUFFIX,
    ),
    **{
        f"nhop_{h}": Dataset(
            id=f"nhop_{h}",
            eval_path=_NHOP_EVAL[h],
            pool_path=_NHOP_POOL,
            scorer=scoring.STRING_INT,
            system_prompt=prompt.NHOP_SYSTEM_PROMPT,
            filler_suffix=prompt.NHOP_FILLER_SUFFIX,
            answer_schema="string",           # n-hop golds are names/phrases as well as numbers
            strict_prompt=prompt.NHOP_STRICT_NOCOT_PROMPT,  # adaptive keeps anti-think, allows phrases
            max_answer_tokens=40,             # clears the longest gold (~14 tokens) with margin
        )
        for h in _NHOP_HOPS
    },
}


MODELS: Dict[str, Model] = {
    # Rates verified live against OpenRouter /models. All three route via the OpenRouter alias
    # namespace rather than a native provider SDK — see the root README's transport note.
    "anthropic/claude-opus-4.5":  Model("anthropic/claude-opus-4.5", "openrouter", None, 5.00, 25.00),
    # Verified live: accepts the reasoning-disable parameter cleanly (4/4 test calls,
    # reasoning_tokens=0, no reasoning content) — same price tier as gpt-5.5, its predecessor here.
    "openai/gpt-5.6-sol":         Model("openai/gpt-5.6-sol", "openrouter", None, 5.00, 30.00),
    # Adaptive-thinking-only: rejects an explicit reasoning-disable parameter under every channel
    # (verified live) — its no-CoT compliance instead rests on either a strict system prompt (the
    # `append` channel, scored wrong if any reasoning is reported) or a forced tool call
    # (`structured` — its default here; `--method append` forces the imperfect natural channel
    # instead, for a controlled comparison). See the root README for the caveat on what a forced
    # tool call does and doesn't prove about internal reasoning.
    "anthropic/claude-fable-5":   Model("anthropic/claude-fable-5", "openrouter", None, 10.00, 50.00),
}


def model_info(model_id: str) -> Model:
    """The registered row, or a synthesized conservative fallback from the id shape — so ad-hoc
    smoke ids keep working without a registry edit (estimates err high rather than failing)."""
    if model_id in MODELS:
        return MODELS[model_id]
    if "/" in model_id:
        return Model(model_id, "openrouter", None, 5.00, 30.00)
    return Model(model_id, "openai", None, 5.00, 30.00)


# The no-CoT channel each model resolves to by default, verified LIVE (not runtime-probed — this
# repo ships no capability-detection infrastructure, just the answer): opus-4.5 accepts a trailing
# assistant message as a genuine continuation (prefill, clean). gpt-5.6-sol accepts the
# reasoning-disable parameter cleanly (4/4 test calls, reasoning_tokens=0) but does NOT genuinely
# support prefill — two probes with a PARTIAL prefix ("Answer: -" on gold -37, "Answer: 3" on
# gold 391) came back as full duplicated answers ("-37", "391") rather than completions ("37",
# "91"), proving it generates a fresh reply rather than continuing the given text — so it
# defaults to append instead. The adaptive-thinking-only model rejects reasoning-disable under
# every channel, so append routes it to the strict-prompt fallback (imperfect — it refuses
# outright at low augmentation), and its default here is instead a FORCED TOOL CALL (`structured`)
# — see the root README for what that channel does and doesn't prove about internal reasoning.
# `--method` always overrides this map for a controlled per-run comparison (e.g. `--method append`
# to see the adaptive model's natural, imperfect channel).
_DEFAULT_METHOD: Dict[str, str] = {
    "anthropic/claude-opus-4.5": "prefill",
    "openai/gpt-5.6-sol": "append",
    "anthropic/claude-fable-5": "structured",
}


def resolve_method(model_id: str, ds: Dataset, forced: Optional[str] = None) -> str:
    """The no-CoT method for one (model, dataset) cell: `--method` (forced) wins, else the
    model's default from `_DEFAULT_METHOD`."""
    if forced:
        return forced
    try:
        return _DEFAULT_METHOD[model_id]
    except KeyError:
        raise ValueError(f"no default method for {model_id!r} — pass --method explicitly") from None
