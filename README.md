# no-cot-harness

A harness for the no-CoT filler-token / problem-repeat replication: does giving a model more
forward-pass compute (repeated copies of a problem, or meaningless filler tokens) improve its
accuracy when forced to answer immediately, with no chain-of-thought? Run across eight models via
the OpenRouter alias namespace (or, for `anthropic/*`, the native API): `anthropic/claude-opus-4.5`,
`openai/gpt-5.6-sol`, `anthropic/claude-fable-5`, `anthropic/claude-opus-5` (the first post) and
`google/gemini-3.1-pro-preview`, `moonshotai/kimi-k3`, `anthropic/claude-fable-5.1`,
`openai/gpt-6-astra` (the second).

Two research updates are built on it:

- [Single Forward Pass Evals on Fable, Opus 5, and GPT-5.6-Sol](https://www.lesswrong.com/posts/bxaWTNrdgJpkLXmgm/single-forward-pass-evals-on-fable-opus-5-and-gpt-5-6-sol)
  — figures in `results/figures/`, run spec `condition_matched_500`.
- Yet another concerning result on Astra's no-CoT capabilities — figures in
  `results/figures_with_astra/` (see the map below), presentation spec `blog1_with_astra` over the
  per-model run specs `astra_blog1` + `astra_nhop4`, `fable51_cm`, `kimi_k3_blog1` + `kimi_k3_nhop4`,
  and `newarm_31pro` (Gemini 3.1 Pro).

Replicates the experiments from two posts by Ryan Greenblatt:

- [Recent LLMs can use filler tokens or problem repeats to improve (no-CoT) math
  performance](https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L/recent-llms-can-use-filler-tokens-or-problem-repeats-to)
  (LessWrong, Dec 2025) — the math filler/repeat experiments.
- [Recent LLMs can do 2-hop and 3-hop latent (no-CoT) reasoning on natural
  facts](https://www.lesswrong.com/posts/aYtrLhoZtCKZnfBvA/recent-llms-can-do-2-hop-and-3-hop-latent-no-cot-reasoning)
  (LessWrong, Jan 2026) — the n-hop fact-composition experiments.

"The source paper" / "the source LessWrong post" in code comments refer to these.

## Condition-matched few-shot demos

Few-shot demos can be shown plainly, or rendered through the same repeat/filler condition as the
query (`--match-demos`) — see `harness/prompt.py`'s `build_messages`.

## Models

Each model's default no-CoT channel is in `harness/registry.py`'s `_DEFAULT_METHOD` map, with the
live evidence behind each choice in the comments there. `--method` overrides any model's default
for a controlled comparison.

## Transport

Every model is reached through OpenRouter by default: one OpenAI-compatible API and one unified
reasoning-disable control across both providers, so all four models run through identical request
code. `anthropic/...` ids can optionally route through the native Anthropic API instead
(`--transport anthropic_native`) — same prompts, same channels, but with genuine prompt caching
(OpenRouter was observed live to never cache these models) and access to the Message Batches API
(`python -m harness.anthropic_batch`, 50% off both input and output).

### What the no-CoT channels do and don't prove

Models that accept an explicit reasoning-disable (`opus-4.5`, `gpt-5.6-sol`, `kimi-k3`) run with
reasoning turned off at the API level. The adaptive-thinking-only models (`fable-5`, `opus-5`)
reject that parameter under every channel, so they default to a forced tool call (`structured`):
the answer must come back as tool input, which makes free-text chain-of-thought in the output
impossible — but forcing the output shape is not, by itself, proof that no internal reasoning pass
occurred.

The mandatory-reasoning endpoints (`gpt-6-astra`, `fable-5.1`, `gemini-3.1-pro-preview`) reject
the disable AND (for Astra and Fable 5.1) the forced tool call, so they run their strict
no-think system prompt through `append` (Astra, Fable 5.1) or `prefill` (Gemini). No effort or
reasoning parameter is sent; the prompt is what suppresses reasoning. A controlled 2x2 on shared
items showed the prompt, not the channel, does the work: under the plain prompt Astra reasoned on
6/6 items through both structured mechanisms, under the strict prompt on 0/6.

Independent of channel, any response that reports reasoning tokens or visible reasoning content is
scored wrong (`scoring.nocot_violation`), with the caveat that some OpenRouter providers omit the
reasoning-token count. Across the 28,260 Astra rows and the 28,260 Fable 5.1 rows behind the
second post, the APIs billed zero reasoning tokens and returned zero reasoning text, and billed
output tokens equal the visible `Answer: X` length. `--method append` runs the adaptive models'
strict-prompt channel instead for a controlled comparison.

### Correction to the first post's GPT-5.6-Sol numbers

Sol's committed n-hop aggregate had been snapshotted mid-collection, between a 403-error window
and its successful retries, so 11 of 36 n-hop cells scored errored items as wrong. The committed
`condition_matched_500_aggregate.json` is rebuilt under the harness's own estimand (last clean row
per cell, errors excluded); the corrected numbers shift by at most +2.05pp, all within the
published CIs, with no ordering changes.

## Data

**No dataset generator or data is included.** `harness/registry.py`'s `eval_path`/`pool_path`
describe where the harness expects to find them locally; building or obtaining them is left to
you — reach out to the maintainers for details. Expected schema (`harness/schema.py`): one JSON
object per line, `{"id", "dataset_id", "problem", "gold_answer", "metadata": {...}}`. Few-shot
pool records additionally need `"gold_answer_str"` — the answer exactly as the demo's
`Answer: X` turn should render it (see `harness/prompt.py`'s `build_messages`).

## Running

```
pip install -r requirements.txt
export OPENROUTER_API_KEY=...

python -m harness.sweep --smoke --n 20                          # live e2e check, plain demos
python -m harness.sweep --smoke --n 20 --match-demos             # live e2e check, condition-matched
python -m harness.sweep --run condition_matched_500 --estimate   # cost table, no submit
python -m harness.sweep --run condition_matched_500 --max-budget-usd 50
python scripts/replay_store.py                                   # $0 regression gate on the store
```

## Layout

- `harness/` — the experiment itself:
  - `config.py` — paths, the model roster, the repeat/filler condition grids, and the named run
    specs the CLI accepts (`--run ...`).
  - `registry.py` — the dataset and model registries, and `resolve_method`, which picks each
    (model, dataset) pair's default no-CoT channel (with the live evidence in comments).
  - `conditions.py` — renders a problem through a condition: repeated N times, or followed by a
    count-to-N filler sequence.
  - `prompt.py` — assembles the few-shot messages around the rendered problem; demos are plain by
    default or rendered through the query's condition with `--match-demos`.
  - `backends.py` — one backend per no-CoT channel (prefill / append / structured) and transport
    (OpenRouter or native Anthropic): builds the request that forces an immediate answer.
  - `sweep.py` — the CLI driver: expands a run spec into cells, runs them concurrently with cost
    estimates and a budget cap, and appends to a resumable JSONL store (rerunning skips
    already-collected cells).
  - `anthropic_batch.py` — the same cells submitted through Anthropic's async Message Batches API
    at its 50% discount, writing rows into the same store.
  - `scoring.py` — answer parsing per answer type, exact-match-after-normalization for string
    golds, and the no-CoT violation rule (any reported reasoning scores the item wrong).
  - `stats.py` — paired bootstrap CIs and paired t-tests with Holm correction.
  - `schema.py` — JSONL record loading.
- `presentation/` — the figure-generation scripts behind everything in `results/figures/` and
  `results/figures_with_astra/`, all reading the sweep store or committed aggregates directly
  (shared helpers in `figures.py`; each script's docstring says which figure it draws). Set
  `SECONDLOOK_WHITE_BG=1` for the white-background variants used in the second post.
  - `plot_hops_curve.py` — accuracy vs. number of hops (second post, Figure 1).
  - `token_matched_slopes.py` — mean provider-reported prompt tokens per condition (and slopes
    per doubling of prompt length) into `results/stats/token_matched_slopes.json`.
  - `plot_prompt_length.py` — the dose-response panels behind the post's interactive picker:
    accuracy or change-vs-baseline against prompt length, filler count, or repeat count.
- `scripts/`
  - `replay_store.py` — re-scores every stored response and asserts it matches the store.
  - `reasoning_ceiling_probe.py` — a one-off control measuring accuracy with reasoning genuinely
    unconstrained, to confirm low no-CoT scores reflect real suppression, not incapability.
- `results/` — committable aggregates/figures (no problem text). `runs/` and `data/` are
  gitignored.

### Figure map for the second post (`results/figures_with_astra/`)

| In the post | File(s) |
|---|---|
| Figure 1 (accuracy vs. hops) | `figure1_hops_best_condition.png` |
| Figure 2 (Comp-Math and 4-Hop, baseline vs. peak) | `figure2_baseline_vs_peak_compmath_4hop.png` |
| Figure 3 (4-Hop, baseline vs. peak) | `baseline_vs_peak/nhop_4.png` |
| Utilization picker (accuracy / change vs. baseline, by dataset and intervention) | `prompt_length/<dataset>__<both\|filler\|repeat>__<acc\|delta>.png` |
| Per-dataset picker | `baseline_vs_peak/*.png` |
| Per-model picker | `baseline_vs_peak_by_model/*.png` |
| Supplementary: per-condition accuracy and change-vs-baseline lines | `lines/*.png`, `delta_lines/*.png` |
| Supplementary: all-dataset baseline-vs-peak bars, hops with every baseline | `supp_baseline_vs_peak_all_datasets.png`, `supp_hops_baseline_vs_best.png` |

Models appear in release order everywhere: Gemini 3.1 Pro, Fable 5, GPT-5.6-Sol, Kimi k3,
Fable 5.1, GPT-6-Astra. Opus 4.5 and Opus 5 are omitted from this set; their results are in the
first post's figures.
