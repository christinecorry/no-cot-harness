# no-cot-harness

A harness for the no-CoT filler-token / problem-repeat replication: does giving a model more
forward-pass compute (repeated copies of a problem, or meaningless filler tokens) improve its
accuracy when forced to answer immediately, with no chain-of-thought? Run across four models —
`anthropic/claude-opus-4.5`, `openai/gpt-5.6-sol`, `anthropic/claude-fable-5`,
`anthropic/claude-opus-5` — via the OpenRouter alias namespace.

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

- `harness/` — `registry.py` (datasets, models, method resolution), `backends.py` (no-CoT
  elicitation per channel), `scoring.py` (parsing + the no-CoT violation rule), `sweep.py` (CLI +
  resumable store), `stats.py` (CIs + significance), `prompt.py` / `conditions.py` (prompt
  assembly), `schema.py` (JSONL loading), `anthropic_batch.py` (native Anthropic Batch API
  transport).
- `presentation/` — figure-generation scripts, all reading the sweep store directly:
  - `figures.py` — shared helpers (house style, aggregate/CI loaders, adaptive-model masking).
  - `plot_baseline_vs_peak.py` — baseline vs peak-repeat vs peak-filler accuracy, one figure per
    dataset, bars grouped by model (the headline significance figure).
  - `plot_baseline_vs_peak_by_model.py` — the same numbers transposed: one figure per model, bars
    grouped by dataset.
  - `plot_figure1.py` — the top-line summary figure: baseline vs each model's peak condition, on
    Gen-Arithmetic and 2-Hop side by side.
  - `plot_delta_lines.py` — accuracy lift (condition minus baseline) across the full repeat/filler
    grid, one line per model.
  - `plot_sanity_check_lines.py` — raw accuracy across the same grid, plus per-condition coverage
    counts (a data-completeness check as much as a figure).
  - `plot_condition_match.py` / `plot_cm_vs_plain.py` — plain-demo vs condition-matched accuracy
    comparisons (the latter reads an external plain-demo store; see its docstring for context).
- `scripts/`
  - `replay_store.py` — re-scores every stored response and asserts it matches the store.
  - `reasoning_ceiling_probe.py` — a one-off control measuring accuracy with reasoning genuinely
    unconstrained, to confirm low no-CoT scores reflect real suppression, not incapability.
- `results/` — committable aggregates/figures (no problem text). `runs/` and `data/` are
  gitignored.
