# no-cot-harness

A harness for the no-CoT filler-token / problem-repeat replication: does giving a model more
forward-pass compute (repeated copies of a problem, or meaningless filler tokens) improve its
accuracy when forced to answer immediately, with no chain-of-thought? Run across four models —
`anthropic/claude-opus-4.5`, `openai/gpt-5.6-sol`, `anthropic/claude-fable-5`,
`anthropic/claude-opus-5` — via the OpenRouter alias namespace.

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

Models that accept an explicit reasoning-disable (`opus-4.5`, `gpt-5.6-sol`) run with reasoning
turned off at the API level. The adaptive-thinking-only models (`fable-5`, `opus-5`) reject that
parameter under every channel, so they default to a forced tool call (`structured`): the answer
must come back as tool input, which makes free-text chain-of-thought in the output impossible —
but forcing the output shape is not, by itself, proof that no internal reasoning pass occurred.
Independent of channel, any response that reports reasoning tokens or visible reasoning content is
scored wrong (`scoring.nocot_violation`), with the caveat that some OpenRouter providers omit the
reasoning-token count. Read the adaptive models' numbers with that in mind; `--method append`
runs their natural strict-system-prompt channel instead for a controlled comparison.

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
- `presentation/` — the figure-generation scripts behind everything in `results/figures/`, all
  reading the sweep store or committed aggregates directly (shared helpers in `figures.py`; each
  script's docstring says which figure it draws).
- `scripts/`
  - `replay_store.py` — re-scores every stored response and asserts it matches the store.
  - `reasoning_ceiling_probe.py` — a one-off control measuring accuracy with reasoning genuinely
    unconstrained, to confirm low no-CoT scores reflect real suppression, not incapability.
- `results/` — committable aggregates/figures (no problem text). `runs/` and `data/` are
  gitignored.
