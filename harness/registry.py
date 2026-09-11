"""Registries: which datasets and models exist, and how a (model × dataset) pair is elicited no-CoT.

Two orthogonal axes, so adding a dataset or a model is one row here:
  - `Dataset` bundles everything dataset-specific the driver needs (paths, scorer, prompts,
    structured answer schema, token cap) — the driver itself stays dataset-agnostic.
  - `Model` consolidates provider + pricing.
  - `resolve_method` picks the no-CoT method per (model, dataset): the cheapest clean channel,
    filtered by capability, overridden where a channel is empirically non-compliant.

A dataset's `scorer` is anything exposing `parse_answer / score / answer_form`; every answer type
lives in `harness.scoring` as a small class, and the driver duck-types on the three methods.

This repo ships four models via the OpenRouter alias namespace — two adaptive-thinking-only
models (cannot have their internal reasoning pass disabled via API parameter) and two models with
the standard prefill/append/structured no-CoT channels — to study condition-matched few-shot
demos. See the root README's "Transport" section for why.

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
    batch_input_rate: Optional[float]   # $/M native Anthropic Batch API input; None = no batch path
    batch_output_rate: Optional[float]  # $/M native Anthropic Batch API output; None = no batch path
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
    # A FIXED 500-item subset of gen_arithmetic — same seeded (42) random-sample convention as
    # gen_arithmetic_1000, saved as its own file so it never drifts if eval_full.jsonl changes.
    "gen_arithmetic_500": Dataset(
        id="gen_arithmetic_500",
        eval_path=config.DATA_DIR / "gen_arithmetic" / "eval_500.jsonl",
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
    # A FIXED 500-item subset of comp_math (full set is 897) — same seeded (42) random-sample
    # convention as gen_arithmetic_1000/500.
    "comp_math_500": Dataset(
        id="comp_math_500",
        eval_path=config.DATA_DIR / "comp_math" / "eval_500.jsonl",
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
    # Rates verified live against OpenRouter /models; matches native Anthropic pricing exactly for
    # both anthropic/* ids (checked against claude.com/pricing) — no OpenRouter markup either way.
    # Batch rates (native Anthropic Batch API only — OpenRouter has no batch endpoint, so gpt-5.6-sol
    # has none): 50% off both directions, confirmed on the same pricing page.
    "anthropic/claude-opus-4.5":  Model("anthropic/claude-opus-4.5", "openrouter", 2.50, 12.50, 5.00, 25.00),
    # Verified live: accepts the reasoning-disable parameter cleanly (4/4 test calls,
    # reasoning_tokens=0, no reasoning content) — same price tier as gpt-5.5, its predecessor here.
    "openai/gpt-5.6-sol":         Model("openai/gpt-5.6-sol", "openrouter", None, None, 5.00, 30.00),
    # Adaptive-thinking-only: rejects an explicit reasoning-disable parameter under every channel
    # (verified live) — its no-CoT compliance instead rests on either a strict system prompt (the
    # `append` channel, scored wrong if any reasoning is reported) or a forced tool call
    # (`structured` — its default here; `--method append` forces the imperfect natural channel
    # instead, for a controlled comparison). See the root README for the caveat on what a forced
    # tool call does and doesn't prove about internal reasoning.
    "anthropic/claude-fable-5":   Model("anthropic/claude-fable-5", "openrouter", 5.00, 25.00, 10.00, 50.00),
    # Same price tier as opus-4.5, verified live against both OpenRouter /models and native
    # claude.com/pricing (no premium for the newer model). Verified live: adaptive-thinking-only
    # (400s on manual `thinking.type.enabled`, like fable-5 — NOT extended-thinking-only like
    # opus-4.5) AND rejects assistant-message prefill outright ("This model does not support
    # assistant message prefill" — a hard 400, not even a soft duplicate-instead-of-continue like
    # gpt-5.6-sol). Defaults to `structured` (forced tool call), same reasoning as fable-5.
    "anthropic/claude-opus-5":    Model("anthropic/claude-opus-5", "openrouter", 2.50, 12.50, 5.00, 25.00),
    # Open-weight model, added 2026-08-05. Rates verified live against OpenRouter /models
    # ($3/$15 per M — Moonshot's own endpoint, the pinned one, prices identically to the
    # default route). No native batch path (not an Anthropic model), so it always runs sync.
    "moonshotai/kimi-k3":         Model("moonshotai/kimi-k3", "openrouter", None, None, 3.00, 15.00),
    # Gemini roster, added 2026-08-07 after the elicitation-suite round. Rates verified live
    # against OpenRouter /models/{id}/endpoints (pinned google-ai-studio stack prices
    # identically to the default). No native batch path via this harness.
    "google/gemini-3-flash-preview": Model("google/gemini-3-flash-preview", "openrouter", None, None, 0.50, 3.00),
    "google/gemini-3.5-flash":       Model("google/gemini-3.5-flash", "openrouter", None, None, 1.50, 9.00),
    "google/gemini-3.1-pro-preview": Model("google/gemini-3.1-pro-preview", "openrouter", None, None, 2.00, 12.00),
    # New-arm roster (2026-08-11): channels probed + tournament-verified by the nocot tool
    # (cards in its local store); rates verified live against the PINNED endpoints (see
    # backends._PROVIDER_PINS — z-ai's first-party stack prices above the discount routes).
    "google/gemini-2.5-pro":      Model("google/gemini-2.5-pro", "openrouter", None, None, 1.25, 10.00),
    "qwen/qwen3.7-flash":         Model("qwen/qwen3.7-flash", "openrouter", None, None, 0.03, 0.13),
    "minimax/minimax-m3":         Model("minimax/minimax-m3", "openrouter", None, None, 0.30, 1.20),
    "z-ai/glm-5.2":               Model("z-ai/glm-5.2", "openrouter", None, None, 1.40, 4.40),
    "nvidia/nemotron-3-ultra-550b-a55b": Model("nvidia/nemotron-3-ultra-550b-a55b", "openrouter", None, None, 0.60, 3.60),
    "meta-llama/llama-4-maverick": Model("meta-llama/llama-4-maverick", "openrouter", None, None, 0.35, 1.15),
    # GPT-6 Astra, added 2026-09-09 (OpenAI's 2026-09-03 frontier release). Rates verified live
    # against OpenRouter /models/openai/gpt-6-astra/endpoints: the pinned first-party "openai"
    # stack is $10/$50 per M (standard tier; the "openai/flex" tag is $5/$25 at lower priority
    # and "openai/fast" is $20/$100 — same weights, different queue). No batch path here.
    "openai/gpt-6-astra":         Model("openai/gpt-6-astra", "openrouter", None, None, 10.00, 50.00),
    # Fable 5.1 (2026-09-10). Same tier as fable-5 on both transports (verified live against
    # OpenRouter /models: $10/$50 sync, :batch $5/$25; native id claude-fable-5-1 listed by
    # /v1/models). Channel: see the probe/matrix cards before any run.
    "anthropic/claude-fable-5.1": Model("anthropic/claude-fable-5.1", "openrouter", 5.00, 25.00, 10.00, 50.00),
}


def model_info(model_id: str) -> Model:
    """The registered row, or a synthesized conservative fallback from the id shape — so ad-hoc
    smoke ids keep working without a registry edit (estimates err high rather than failing)."""
    if model_id in MODELS:
        return MODELS[model_id]
    if "/" in model_id:
        return Model(model_id, "openrouter", None, None, 5.00, 30.00)
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
    # Adaptive-thinking-only (like fable-5) AND hard-rejects prefill outright ("This model does
    # not support assistant message prefill", native API) — same default as fable-5, same reason.
    "anthropic/claude-opus-5": "structured",
    # Gemini defaults from the 2026-08-07 suite + matrix cards. 3-flash-preview: full clean
    # surface, append is the cheapest clean channel (disable obeyed — api_enforced).
    # 3.1-pro-preview and 3.5-flash: mandatory-reasoning endpoints whose ONLY fully-clean
    # channel is the prefill continuation (suite: prefill 1.00 clean on both; the real matrix
    # decided prefill/clean for 3.1-pro — 24/24 immediate, 0 violations, observed_only
    # enforcement since no disable knob backs it).
    "google/gemini-3-flash-preview": "append",
    "google/gemini-3.5-flash": "prefill",
    "google/gemini-3.1-pro-preview": "prefill",
    # kimi-k3 accepts the unified reasoning-disable and OBEYS it (verified live 2026-08-05:
    # probe clean, and a 240-call condition-matched check across all four datasets returned
    # 219/219 immediate answers with 0 reasoning tokens and 0 reasoning content — including
    # nhop_3, where the adaptive Claude models write prose reasoning instead). Prefill is not
    # available (the harness gates it to Claude ids), so append is both the cheapest clean
    # channel and the paper-faithful one. Requests are pinned to Moonshot's own serving stack
    # (backends._PROVIDER_PINS) — the compliance evidence must describe ONE stack.
    "moonshotai/kimi-k3": "append",
    # New-arm defaults from the nocot tournaments (2026-08-10/11): append-with-disable is the
    # clean channel for the four obeyed-knob open-weight models. gemini-2.5-pro is
    # mandatory-reasoning; its default is the closed-think prefill (see
    # backends.OpenRouterClosedThinkPrefillBackend) — zero violations on every probed cell,
    # where its OPEN prefill variants (plain and strict both) let the mandatory thinking pass
    # re-fire on large gen filler. llama-4-maverick: append is clean on nhop, but its math
    # append pilots wrote visible prose CoT on ~3-5% of rows (up to ~15% in the worst cells)
    # while its structured channel (strict json_schema — not a tool call) tested 16/16 clean;
    # that split is encoded per-dataset in _DATASET_METHOD_OVERRIDES below.
    "google/gemini-2.5-pro": "prefill_closed",
    "qwen/qwen3.7-flash": "append",
    "minimax/minimax-m3": "append",
    "z-ai/glm-5.2": "append",
    "nvidia/nemotron-3-ultra-550b-a55b": "append",
    "meta-llama/llama-4-maverick": "append",
    # gpt-6-astra (probe + channel matrix 2026-09-09): a
    # MANDATORY-reasoning endpoint, unlike gpt-5.6-sol — the unified disable 400s on OpenRouter
    # and the native API accepts only effort low/medium/high/xhigh (no "none"/"minimal"). So
    # `append` routes to the adaptive strict-prompt backend (backends._MANDATORY_REASONING),
    # which was the ONLY clean channel on the gen_arithmetic matrix: 9/9 immediate, 0 billed
    # reasoning tokens across baseline/r5/f300. The STRICT PROMPT is what does the work, not
    # the channel: a controlled 2x2 on the same six gen items (2026-09-10) had BOTH structured
    # mechanisms (forced tool and json_schema) bill 15-84 reasoning tokens on 6/6 under the
    # dataset's plain prompt and 0/6 under the strict prompt. The structured backends omit the
    # strict prompt by design, which is why their matrix cells were dirty; prefill restated the
    # "Answer:" cue on 8/9 (scored reasoning_first). observed_only enforcement: re-pilot
    # (--n 20) before every full submission. n=3 per matrix cell — thin; the pilot is the gate.
    "openai/gpt-6-astra": "append",
    # fable-5.1 (probed + native matrix 2026-09-10): adaptive-only (thinking disabled/enabled
    # both 400; effort "low" is the floor), prefill hard-400s, and tool_choice "tool"/"any" is
    # no longer supported — so fable-5's forced-tool structured channel does not exist here.
    # Native append = strict prompt + thinking adaptive + effort low
    # (backends._NATIVE_ADAPTIVE_THINKING): 48/48 immediate, 0 thinking blocks, output tokens
    # == answer length on gen_arithmetic_500 and nhop_2 x {baseline, r5, f300}. Under the
    # dataset's plain prompt the same channel wrote in-text CoT on 6/8 gen items, so the
    # strict prompt is the enforcement (same finding as astra). Batch path:
    # `python -m harness.anthropic_batch --submit --run fable51_cm --models anthropic/claude-fable-5.1`.
    "anthropic/claude-fable-5.1": "append",
}


# Per-(model, dataset) overrides beat _DEFAULT_METHOD: card evidence that a model's clean
# channel differs BY DATASET, encoded here so a named run cannot silently collect a dirty
# channel when the operator omits --method. An explicit --method still beats both, keeping
# forced controlled comparisons possible.
_DATASET_METHOD_OVERRIDES: Dict[tuple, str] = {
    ("meta-llama/llama-4-maverick", "gen_arithmetic_500"): "structured",
    ("meta-llama/llama-4-maverick", "comp_math_500"): "structured",
}


def resolve_method(model_id: str, ds: Dataset, forced: Optional[str] = None) -> str:
    """The no-CoT method for one (model, dataset) cell: `--method` (forced) wins, then
    `_DATASET_METHOD_OVERRIDES`, else the model's default from `_DEFAULT_METHOD`."""
    if forced:
        return forced
    override = _DATASET_METHOD_OVERRIDES.get((model_id, ds.id))
    if override:
        return override
    try:
        return _DEFAULT_METHOD[model_id]
    except KeyError:
        raise ValueError(f"no default method for {model_id!r} — pass --method explicitly") from None
