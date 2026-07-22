# no-cot-harness

A harness for the no-CoT filler-token / problem-repeat replication: does giving a model more
forward-pass compute (repeated copies of a problem, or meaningless filler tokens) improve its
accuracy when forced to answer immediately, with no chain-of-thought? Run across three models —
`anthropic/claude-opus-4.5`, `openai/gpt-5.6-sol`, `anthropic/claude-fable-5` — via the OpenRouter
alias namespace.

## Condition-matched few-shot demos

Few-shot demos can be shown plainly, or rendered through the same repeat/filler condition as the
query (`--match-demos`) — see `harness/prompt.py`'s `build_messages`.

## Models

Each model's default no-CoT channel is in `harness/registry.py`'s `_DEFAULT_METHOD` map, with the
live evidence behind each choice in the comments there. `--method` overrides any model's default
for a controlled comparison.

## Data

**No dataset generator or data is included.** `harness/registry.py`'s `eval_path`/`pool_path`
describe where the harness expects to find them locally; building or obtaining them is left to
you — reach out to the maintainers for details. Expected schema (`harness/schema.py`): one JSON
object per line, `{"id", "dataset_id", "problem", "gold_answer", "metadata": {...}}`.

## Running

```
pip install -r requirements.txt
export OPENROUTER_API_KEY=...

python -m harness.sweep --smoke --n 20                       # live e2e check, plain demos
python -m harness.sweep --smoke --n 20 --match-demos          # live e2e check, condition-matched
python -m harness.sweep --run condition_matched --estimate    # cost table, no submit
python -m harness.sweep --run condition_matched --max-budget-usd 50
python scripts/replay_store.py                                # $0 regression gate on the store
```

## Layout

- `harness/` — `registry.py` (datasets, models, method resolution), `backends.py` (no-CoT
  elicitation per channel), `scoring.py` (parsing + the no-CoT violation rule), `sweep.py` (CLI +
  resumable store), `stats.py` (CIs + significance), `prompt.py` / `conditions.py` (prompt
  assembly), `schema.py` (JSONL loading).
- `presentation/` — `figures.py` (shared helpers), `plot_condition_match.py` (plain vs
  condition-matched accuracy).
- `scripts/replay_store.py` — re-scores every stored response and asserts it matches the store.
- `results/` — committable aggregates/figures (no problem text). `runs/` and `data/` are
  gitignored.
