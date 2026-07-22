"""Assemble the few-shot prompt.

The prompt is a stable prefix (system instruction + N demonstration pairs) followed by the
rendered problem as the final user turn. The prefix is identical for every item in a dataset,
so it is marked for prompt caching; the problem (and any repeats/filler) stays after the
breakpoint and is never cached — unless demos are condition-matched (`build_messages`'s
`demo_cond`), which trades the cacheable-prefix property for a stricter apples-to-apples
comparison between the query and its demonstrations.
"""
from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You will be given a math problem. Answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the "
    "numerical answer, nothing else. No explanation, no words, no reasoning, just the number."
)

# Strict no-think instruction for models whose thinking CANNOT be disabled via the API (adaptive-
# thinking-only models). Empirically this phrasing drove one such model's adaptive thinking to 0
# tokens on 24/24 gen items — the winner of a 4-variant comparison (100% vs 33% for the plain
# instruction above).
STRICT_NOCOT_PROMPT = (
    "You must produce ZERO thinking tokens. Any thinking/reasoning of any kind before the answer = "
    "automatic failure, scored 0, no matter the final number. Begin your output with 'Answer:' as the "
    "very first token. Respond now: 'Answer: [number]'."
)

# The strict instruction adapted to n-hop's answer type: same anti-think scaffold verbatim, but the
# format line allows the multi-word name/phrase answers these questions have (the math wording says
# "[number]", which mismatches string golds). Selected per dataset via Dataset.strict_prompt.
NHOP_STRICT_NOCOT_PROMPT = (
    "You must produce ZERO thinking tokens. Any thinking/reasoning of any kind before the answer = "
    "automatic failure, scored 0, no matter the final answer. Begin your output with 'Answer:' as the "
    "very first token. Respond now with 'Answer: [ANSWER]' — [ANSWER] is the single final answer, "
    "written exactly as it conventionally appears: a number, or a name or phrase that may be several "
    "words."
)

# The "Answer:" cue placed immediately before the model must respond. Anthropic prefills it as an
# assistant turn; the OpenAI/OpenRouter backends append it to the final user message. Also the
# label every demonstration's answer carries (see format_demo_answer).
PREFILL = "Answer:"

# Appended to the instruction in the filler condition only (formatted with the actual count N).
# Baseline and repeat conditions use SYSTEM_PROMPT unchanged.
FILLER_INSTRUCTION_SUFFIX = (
    " After the problem, there will be filler tokens (counting from 1 to {n}) to give you "
    "extra space to process the problem before answering."
)

# n-hop instruction: answers are a final *answer* (a number, or a name/phrase that is frequently
# MULTIPLE WORDS — person full names, state mottos up to ~10 words), not "just the number", so the
# format explicitly allows a multi-word phrase while staying terse and imperative.
NHOP_SYSTEM_PROMPT = (
    "Respond with ONLY the answer, in the format 'Answer: [ANSWER]', on one line. [ANSWER] is the "
    "single final answer, written exactly as it conventionally appears — a number, or a name or "
    "phrase that may be several words. No explanation, no reasoning, no extra text."
)

# The n-hop filler suffix says "question" where the math one says "problem".
NHOP_FILLER_SUFFIX = (
    " After the question, there will be filler tokens (counting from 1 to {n}) to give you extra "
    "space to process the question before answering."
)


def format_demo_answer(gold_str: str) -> str:
    """A demonstration's assistant turn, e.g. "Answer: 42" — identical across all backends."""
    return f"{PREFILL} {gold_str}"


def system_for(cond, base: str = SYSTEM_PROMPT, suffix: str = FILLER_INSTRUCTION_SUFFIX) -> str:
    """The system instruction for a condition.

    Filler conditions append the filler explanation (with the real token count); baseline and
    repeat conditions use the base instruction unchanged — matching the paper, which adds the
    explanatory sentence only when filler tokens follow. `base` lets a backend swap in a different
    instruction (e.g. STRICT_NOCOT_PROMPT for adaptive-thinking models); `suffix` lets a dataset
    swap in its own filler wording (e.g. the n-hop "question" phrasing) — both keep the same rule.
    """
    if getattr(cond, "kind", None) == "filler" and cond.value > 0:
        return base + suffix.format(n=cond.value)
    return base


def build_messages(
    pool: List[Dict[str, Any]],
    problem_text: str,
    backend: Any,
    *,
    demo_cond: Any = None,
) -> List[Dict[str, Any]]:
    """Build the message list: demonstration pairs (cached prefix) + the final problem turn.

    Demonstrations are identical across providers (each assistant turn is "Answer: N"); the
    backend only decides whether the few-shot prefix carries an Anthropic cache-control marker
    (`wants_cache_control`; OpenAI/OpenRouter cache automatically). How the FINAL answer is
    elicited (prefill vs appending "Answer:") is the backend's job, in `build_params`.

    `demo_cond`: None (default) shows every demo as the bare, unaugmented problem — the
    paper-faithful default (only the query gets repeats/filler). Passing a `Condition` renders
    EACH demo through it too (condition-MATCHED few-shot) — an explicit opt-in variant, never the
    default, since it changes the cached-prefix shape per condition and multiplies prompt length
    for augmented conditions.
    """
    from . import conditions  # local import: avoid a hard dependency for callers that never match
    messages: List[Dict[str, Any]] = []
    last = len(pool) - 1
    for i, ex in enumerate(pool):
        demo_text = conditions.render(ex["problem"], demo_cond) if demo_cond is not None else ex["problem"]
        messages.append({"role": "user", "content": demo_text})
        answer = format_demo_answer(ex["gold_answer_str"])
        if i == last and backend.wants_cache_control:
            # Cache breakpoint on the last demonstration: caches system + all demonstrations.
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer, "cache_control": {"type": "ephemeral"}}],
            })
        else:
            messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": problem_text})
    return messages
