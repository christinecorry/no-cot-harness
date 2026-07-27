"""One-off control: how accurate are these models when reasoning is genuinely UNCONSTRAINED (no
no-CoT enforcement of any kind), scored by loose substring match rather than the harness's strict
exact-match `Answer:` parser. This is a sanity/ceiling check on raw capability, NOT part of the
no-CoT replication sweep — it exists to confirm these models can actually solve the problems when
allowed to think, so a low no-CoT score means "genuinely suppressed," not "can't do math."

Zero-shot: no few-shot demos, no repeat/filler augmentation, no reasoning-disable param, no
`Answer:` append/prefill, no forced tool call — literally just the problem text and a plain
instruction. All calls go through OpenRouter (transport policy), with the Anthropic provider pin
still applied to anthropic/* ids (pin is a transport concern, independent of reasoning being on).

Items are a fixed, reproducible random sample (seed 42, matching the project's existing
seeded-subsample convention) drawn from each dataset's real eval file.

Per-item rows (contain problem text) go to runs/reasoning_ceiling_probe.jsonl — gitignored, never
committed. Only the aggregate summary (accuracy counts, no problem text) is written to
results/reasoning_ceiling_probe_summary.json, which is committable.

    python scripts/reasoning_ceiling_probe.py --models anthropic/claude-fable-5 --dataset gen_arithmetic_500 --n 2   # pilot
    python scripts/reasoning_ceiling_probe.py   # full run: 4 models x 5 datasets x 20 items
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness import config, registry, schema  # noqa: E402
from harness.backends import OPENROUTER_BASE_URL, _ANTHROPIC_PROVIDER_PIN  # noqa: E402

DATASETS = ["gen_arithmetic_500", "comp_math_500", "nhop_2", "nhop_3", "nhop_4"]
MODELS = ["anthropic/claude-opus-4.5", "openai/gpt-5.6-sol", "anthropic/claude-fable-5",
          "anthropic/claude-opus-5"]
SEED = 42
SYSTEM_PROMPT = ("Solve the following problem. Reason through it however you like, then finish "
                 "your response with a line in EXACTLY this form, and nothing after it:\n"
                 "Final Answer: <your answer>")
_FINAL_ANSWER_MARKER = re.compile(r"final answer\s*:\s*", re.IGNORECASE)
OUT_PATH = config.RUNS_DIR / "reasoning_ceiling_probe.jsonl"
SUMMARY_PATH = config.RESULTS_DIR / "reasoning_ceiling_probe_summary.json"


def sample_items(dataset_id: str, n: int, seed: int) -> List[Dict[str, Any]]:
    items = schema.load_jsonl(registry.DATASETS[dataset_id].eval_path)
    return random.Random(seed).sample(items, n)


def build_client():
    import openai
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ.get("OPENROUTER_API_KEY"),
                         default_headers={"X-Title": "no-cot-harness-reasoning-probe"},
                         timeout=120.0, max_retries=2)


def call_model(client: Any, model: str, problem: str, max_tokens: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}],
        "max_completion_tokens": max_tokens,
    }
    if model.startswith("anthropic/"):
        params["extra_body"] = {"provider": dict(_ANTHROPIC_PROVIDER_PIN)}
    resp = client.chat.completions.create(**params)
    if getattr(resp, "choices", None) in (None, []):
        err = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
        raise RuntimeError(f"provider returned no choices: {err or 'unknown error'}")
    msg = resp.choices[0].message
    text = msg.content or ""
    reasoning = (getattr(msg, "reasoning", None)
                or (getattr(msg, "model_extra", None) or {}).get("reasoning") or "")
    u = resp.usage
    cdetails = getattr(u, "completion_tokens_details", None)
    reasoning_tokens = (getattr(cdetails, "reasoning_tokens", 0) or 0) if cdetails else 0
    return {
        "text": text, "reasoning_chars": len(reasoning), "reasoning_tokens": reasoning_tokens,
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
    }


_INT_WORD_BOUNDARY = r"(?<!\d){}(?!\d)"

# String-answer normalization, reimplemented from scratch after reading (not copying) the
# n-hop paper's own eval_multi_hop.py `normalize_answer`/`check_answer` (one-off read exception,
# approved 2026-07-27 — see PROGRESS.md; only the two generator files are vendored/copied, this
# scoring logic is NOT). Their real, ACTIVE scoring is exact-match-after-normalization — a
# generic substring/subsequence check appears in their file but is commented out and unused —
# so this mirrors their normalization rules rather than a looser fuzzy match.
#
# The reference data below (person-name aliases, state mottos/flowers) is plain factual reference
# data, not code — the same category as our own vendored generator's fact tables, which this
# project already commits (harness/data_curation/vendor/multi_hop/inputs/ in the sibling private
# repo). Hardcoded here rather than imported since this public repo has no vendored copy of that
# generator to import from.
_SUFFIXES = {"jr", "sr", "i", "ii", "iii", "iv", "v"}

# Historical figures where the generic first+last-word rule picks the WRONG two words — each is
# conventionally known by a middle+last or other non-first+last form. Checked against our own
# nhop eval files: 3 of these 8 actually appear as gold answers (Boyd Orr, Cremer, Debye); the
# rest are included for full parity since they're the same Nobel-laureate fact domain.
_NAME_ALIASES = {
    "robert bruce merrifield": "bruce merrifield",
    "oscar arias sanchez": "oscar arias",
    "john boyd orr": "boyd orr",
    "hermann emil fischer": "emil fischer",
    "petrus debye": "peter debye",
    "adolf otto reinhold windaus": "adolf windaus",
    "william randal cremer": "randal cremer",
    "rigoberta menchu tum": "rigoberta menchu",
}
# US state mottos/flowers (all 50 states + DC) — confirmed several appear as gold answers in our
# nhop eval files (e.g. Pennsylvania's motto). These are non-name multi-word phrases that
# `_remove_middle_word` must NOT touch (it would mangle e.g. Colorado's 3-word Latin motto by
# treating it as a person's name and dropping the middle word) — mirrors the original's
# `NORMALIZED_NON_NAME_SET` guard.
_STATE_MOTTOS = [
    "We Dare Defend Our Rights", "North to the Future", "Ditat Deus", "Regnat Populus", "Eureka",
    "Nil Sine Numine", "Qui Transtulit Sustinet", "Liberty and Independence", "In God We Trust",
    "Wisdom, Justice, and Moderation", "Ua Mau ke Ea o ka Aina i ka Pono", "Esto Perpetua",
    "State Sovereignty, National Union", "The Crossroads of America",
    "Our Liberties We Prize and Our Rights We Will Maintain", "Ad Astra per Aspera",
    "United We Stand, Divided We Fall", "Union, Justice, and Confidence", "Dirigo",
    "Fatti Maschii, Parole Femine", "Ense Petit Placidam Sub Libertate Quietem",
    "Si Quaeris Peninsulam Amoenam Circumspice", "L'Étoile du Nord", "Virtute et Armis",
    "Salus Populi Suprema Lex Esto", "Oro y Plata", "Equality Before the Law",
    "All for Our Country", "Live Free or Die", "Liberty and Prosperity", "Crescit Eundo",
    "Excelsior", "Esse Quam Videri", "Liberty and Union Now and Forever, One and Inseparable",
    "With God All Things Are Possible", "Labor Omnia Vincit", "Alis Volat Propriis",
    "Virtue, Liberty, and Independence", "Hope", "Dum Spiro Spero",
    "Under God the People Rule", "Agriculture and Commerce", "Friendship", "Industry",
    "Freedom and Unity", "Sic Semper Tyrannis", "Al-ki", "Montani Semper Liberi", "Forward",
    "Equal Rights", "Justitia Omnibus",
]
_STATE_FLOWERS = [
    "Camellia", "Forget-me-not", "Saguaro Cactus Blossom", "Apple Blossom", "California Poppy",
    "Rocky Mountain Columbine", "Mountain Laurel", "Peach Blossom", "Orange Blossom",
    "Cherokee Rose", "Hawaiian Hibiscus", "Syringa", "Violet", "Peony", "Wild Rose", "Sunflower",
    "Goldenrod", "Magnolia", "White Pine Cone and Tassel", "Black-Eyed Susan", "Mayflower",
    "Pink and White Lady's Slipper", "White Hawthorn Blossom", "Bitterroot", "Sagebrush",
    "Purple Lilac", "Common Meadow Violet", "Yucca Flower", "Rose", "Dogwood",
    "Wild Prairie Rose", "Scarlet Carnation", "Oklahoma Rose", "Oregon Grape",
    "Yellow Jessamine", "Pasque Flower", "Iris", "Bluebonnet", "Sego Lily", "Red Clover",
    "American Dogwood", "Coast Rhododendron", "Rhododendron", "Wood Violet",
    "Indian Paintbrush", "American Beauty Rose",
]
_FLOWER_ALIASES = {
    "hawaiian hibiscus": "hibiscus",
    "white hawthorn blossom": "hawthorn",
    "common meadow violet": "violet",
    "yucca flower": "yucca",
}
_DIRECTIONAL_PREFIXES = ("northern ", "western ", "eastern ", "southern ", "american ")
_HONORIFIC_PREFIXES = ("mr. ", "sir. ", "mr ", "sir ", "lord ")


def _remove_middle_word(s: str) -> str:
    """First + last word for a 3-4 word all-alphabetic phrase (a middle name/initial), preserving
    a trailing suffix (Jr/Sr/I-V) — same narrow scope as the original's `remove_middle_names`:
    a 4-word phrase WITHOUT a suffix is left unchanged (their function only strips a single
    inserted word, not two)."""
    words = s.split()
    if not (3 <= len(words) <= 4):
        return s
    if not all(w.isalpha() for w in words):
        return s
    if words[-1] in _SUFFIXES:
        return f"{words[0]} {words[-2]} {words[-1]}" if len(words) == 4 else s
    if len(words) == 4:
        return s
    return f"{words[0]} {words[2]}"


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _base_normalize(s: str) -> str:
    """Every step EXCEPT the final middle-word removal — used both directly and to build
    `_NON_NAME_SET` (which must be compared against pre-middle-word-removal strings, matching
    the original's `skip_middle_name_normalization=True` when building its own exclusion set)."""
    s = _strip_accents(s.strip().lower())
    for prefix in _HONORIFIC_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r"\s*\([^)]*\)", "", s)  # parenthetical explanation, e.g. "X (also known as Y)"
    for alias, canon in _NAME_ALIASES.items():
        s = s.replace(alias, canon)
    # Washington DC and Alabama's motto each have two equally-valid forms in general use
    # (place name vs "District of Columbia"; English translation vs the original Latin) —
    # canonicalize both directions so either form matches the other.
    s = s.replace("washington, d.c.", "district of columbia")
    s = s.replace("d.c.", "district of columbia")
    if s == "washington":
        s = "district of columbia"
    s = s.replace("audemus jura nostra defendere", "we dare defend our rights")
    for ch in ",.-'`":
        s = s.replace(ch, "")
    s = re.sub(r"\band\b", "", s)  # drop the word "and" -> Oxford-comma-insensitive
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    for prefix in _DIRECTIONAL_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    for alias, canon in _FLOWER_ALIASES.items():
        s = s.replace(alias, canon)
    return s.strip()


_NON_NAME_SET = {_base_normalize(x) for x in _STATE_MOTTOS + _STATE_FLOWERS}


def _normalize_string(s: str) -> str:
    """Full normalization including middle-word removal, EXCEPT for known non-name multi-word
    phrases (state mottos/flowers) where removing a "middle word" would corrupt a legitimate
    answer rather than strip an inserted name — mirrors the original's `NORMALIZED_NON_NAME_SET`
    guard exactly."""
    base = _base_normalize(s)
    if base in _NON_NAME_SET:
        return base
    return _remove_middle_word(base)


def _bounded_gap_pattern(words: List[str]) -> str:
    """Regex requiring `words` in order, allowing at most ONE extra token between each pair —
    a single inserted middle name/word, matching `_remove_middle_word`'s own narrow scope,
    applied here as a bounded search within a response segment (rather than an unbounded
    subsequence match, which could match unrelated text) since our segment isn't a single
    already-isolated answer the way the original's `predicted` string is."""
    escaped = [re.escape(w) for w in words]
    gap = r"(?:\s+\S+)?\s+"
    return r"\b" + escaped[0] + r"\b" + "".join(f"{gap}\\b{w}\\b" for w in escaped[1:])


def _substring_match(segment: str, gold_answer: Any, answer_schema: str) -> bool:
    if answer_schema == "integer" or isinstance(gold_answer, int):
        pattern = _INT_WORD_BOUNDARY.format(re.escape(str(gold_answer)))
        return re.search(pattern, segment) is not None
    # Gold gets the FULL normalization (including the guarded middle-word removal) since it's
    # already an isolated canonical phrase. The response segment only gets the base pass — the
    # 3-4-word middle-word heuristic isn't meaningful applied to a longer, multi-sentence blob.
    gold_norm = _normalize_string(str(gold_answer))
    text_norm = _base_normalize(segment)
    if gold_norm in text_norm:
        return True
    gold_words = gold_norm.split()
    if len(gold_words) >= 2:
        return re.search(_bounded_gap_pattern(gold_words), text_norm) is not None
    return False


def matches_gold(response_text: str, gold_answer: Any, answer_schema: str) -> tuple[bool, str]:
    """Score against the segment AFTER the last "Final Answer:" marker only — never the whole
    response. A blind whole-text search would false-positive on gold values that show up as an
    intermediate step or a value the model considered and rejected mid-reasoning, not its actual
    final answer (the system prompt asks for the marker precisely so this segment is well-defined
    rather than guessed).

    Returns (correct, method): method is "marker" when the marker was found (the intended,
    trustworthy path) or "fallback_last_200_chars" when it wasn't (the model didn't follow the
    format) — logged per-row so a spike in fallback usage is visible rather than silently
    blended into the accuracy number."""
    if not response_text:
        return False, "empty"
    matches = list(_FINAL_ANSWER_MARKER.finditer(response_text))
    if matches:
        segment = response_text[matches[-1].end():]
        return _substring_match(segment, gold_answer, answer_schema), "marker"
    # No marker found (model didn't follow the format) — fall back to a bounded tail window
    # rather than the full text, so a stray earlier mention still can't inflate the score.
    segment = response_text[-200:]
    return _substring_match(segment, gold_answer, answer_schema), "fallback_last_200_chars"


def process_item(client: Any, model: str, dataset_id: str, item: Dict[str, Any], ds: Any,
                 max_tokens: int) -> Dict[str, Any]:
    r = None
    last_err: Exception | None = None
    # In-body provider errors (e.g. "Overloaded" 503) arrive as HTTP 200, so the SDK's own retry
    # logic never sees them — retry a couple times ourselves rather than losing an item to a
    # known-transient blip.
    for attempt in range(3):
        try:
            r = call_model(client, model, item["problem"], max_tokens)
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    if r is None:
        return {"model": model, "dataset": dataset_id, "item_id": item["id"], "error": str(last_err)}
    ok, score_method = matches_gold(r["text"], item["gold_answer"], ds.answer_schema)
    return {"model": model, "dataset": dataset_id, "item_id": item["id"],
           "gold_answer": item["gold_answer"], "correct": ok, "score_method": score_method,
           "error": None, **r}


def load_done_keys(out_path: Path) -> Set[Tuple[str, str, str]]:
    """(model, dataset, item_id) already recorded WITHOUT error — safe to skip on resume. A row
    that previously errored is deliberately not counted as done, so a resumed run retries it."""
    done: Set[Tuple[str, str, str]] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") is None:
                done.add((row["model"], row["dataset"], row["item_id"]))
    return done


def cell_stats(out_path: Path, model: str, dataset_id: str,
               item_ids: Set[str]) -> Tuple[int, int, int, int]:
    """(n, correct, errors, fallback_count) for exactly `item_ids`, deduped by item_id (a resumed
    run can leave both an old error row and a later success row for the same item — the later
    row in file order wins, since jsonl append order is chronological)."""
    latest: Dict[str, Dict[str, Any]] = {}
    with out_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model") != model or row.get("dataset") != dataset_id:
                continue
            if row["item_id"] not in item_ids:
                continue
            latest[row["item_id"]] = row
    correct = sum(1 for r in latest.values() if r.get("error") is None and r["correct"])
    errors = sum(1 for r in latest.values() if r.get("error") is not None)
    fallback = sum(1 for r in latest.values()
                   if r.get("error") is None and r.get("score_method") != "marker")
    return len(latest), correct, errors, fallback


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated model ids; default: all 4")
    ap.add_argument("--dataset", action="append", help="dataset id (repeatable); default: all 5")
    ap.add_argument("--n", type=int, default=20, help="items per dataset (default 20)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent in-flight calls per (model, dataset) cell (default 3, "
                         "matching the project's OpenRouter convention)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    models = args.models.split(",") if args.models else MODELS
    datasets = args.dataset or DATASETS
    client = build_client()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = load_done_keys(out_path)
    if done_keys:
        print(f"resuming: {len(done_keys)} items already completed in {out_path}")

    write_lock = threading.Lock()
    results = []
    with out_path.open("a") as f:
        for dataset_id in datasets:
            ds = registry.DATASETS[dataset_id]
            items = sample_items(dataset_id, args.n, args.seed)
            item_ids = {it["id"] for it in items}
            for model in models:
                todo = [it for it in items if (model, dataset_id, it["id"]) not in done_keys]
                if todo:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        futures = [pool.submit(process_item, client, model, dataset_id, it, ds,
                                              args.max_tokens) for it in todo]
                        for fut in as_completed(futures):
                            row = fut.result()
                            with write_lock:
                                f.write(json.dumps(row) + "\n")
                                f.flush()
                n, correct, errors, fallback_count = cell_stats(out_path, model, dataset_id, item_ids)
                n_scored = n - errors
                acc = correct / n_scored if n_scored else float("nan")
                print(f"{model:28s} {dataset_id:20s} {correct}/{n_scored} = {acc:.1%}"
                     + (f"  ({errors} errors)" if errors else "")
                     + (f"  ({fallback_count} no-marker fallback)" if fallback_count else ""))
                results.append({"model": model, "dataset": dataset_id, "n": n,
                               "n_scored": n_scored, "correct": correct, "acc": acc,
                               "fallback_count": fallback_count, "errors": errors})

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(SUMMARY_PATH).write_text(json.dumps(
        {"seed": args.seed, "n_per_dataset": args.n, "results": results}, indent=2) + "\n")
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
