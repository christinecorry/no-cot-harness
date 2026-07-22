"""All dataset-dependent answer parsing and scoring, in one place.

Every scorer exposes the 3-method surface the driver dispatches on (`parse_answer`, `score`,
`answer_form`) and shares the IMMEDIATE-answer stance that defines this study: the answer must
LEAD the response. A model that reasons first scores wrong even if it lands the value later —
a scorer that walked backward to the last answer-looking line would reward chain-of-thought and
defeat the no-CoT measurement, so every scorer here reads the first line only.

  - IntegerScorer     the math rule (Gen-Arithmetic / competition math): the integer after
                      "Answer:" (or leading the response), exact-match against an integer gold.
  - StringIntScorer   the n-hop fact-composition rule: mixed string/int golds; integer golds match
                      the integer anchored right after "Answer:", string golds match normalized-
                      exactly (no fuzzy/alias matching).

This module also owns the ONE definition of the no-CoT violation rule (`nocot_violation`) and the
record re-scoring recipe (`rescore_record`) shared by the sweep driver and the replay regression
gate — a rule change lands everywhere at once or nowhere.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# --- Unicode normalization (applied ahead of every parser) ----------------------------------------
# Models emit typographic variants of ASCII characters — U+2212 minus signs, no-break/thin spaces,
# curly quotes and apostrophes, en/em dashes, fullwidth digits — that would otherwise fail regexes
# and exact-match comparison written for ASCII. Both the model's answer and (for string-compared
# golds) the gold pass through the same mapping, so the comparison stays symmetric. Accents are
# deliberately NOT folded ("Málaga" != "Malaga"): fuzzy matching is out of scope here, and an
# accent difference is a wrong answer, not a formatting variant.
_UNICODE_TABLE = {
    0x2212: "-",   # minus sign
    0x2013: "-",   # en dash
    0x2014: "-",   # em dash
    0x00A0: " ",   # no-break space
    0x2007: " ",   # figure space
    0x2009: " ",   # thin space
    0x202F: " ",   # narrow no-break space
    0x2018: "'",   # left single quote
    0x2019: "'",   # right single quote / curly apostrophe
    0x201C: '"',   # left double quote
    0x201D: '"',   # right double quote
}
_UNICODE_TABLE.update({0xFF10 + i: str(i) for i in range(10)})  # fullwidth digits ０-９


def _normalize_unicode(text: str) -> str:
    """Map typographic variants to ASCII (see table above); everything else unchanged."""
    return text.translate(_UNICODE_TABLE)


# A leading integer immediately followed by one of these is the FIRST OPERAND of a chain-of-thought
# expression ("-4 % 96 = 92 …"), NOT an answer — shared by both first-line integer rules below.
_OPERATOR_CHARS = set("+-*/%=")


def _is_expression_tail(after: str) -> bool:
    """True when the text after a leading integer makes that integer the first OPERAND of an
    expression — an operator WITH content after it ("% 96 = …", "/3"). A bare trailing operator
    at end of input ("28%", "612-") is formatting/punctuation, not an expression: rejecting it
    would misread a percent-formatted answer as chain-of-thought. Off-format-but-parsable values
    like "2/3" or "175/6" remain rejected — the integer rule must not truncate a fraction to its
    numerator."""
    return bool(after) and after[0] in _OPERATOR_CHARS and bool(after[1:].strip())


# --- the math rule (Gen-Arithmetic / competition math) ---------------------------------------------

# "Answer:" / "Answer =" at the start, then the integer (trailing PROSE after the number is fine;
# a trailing operator is rejected by the shared guard in parse_answer — see _OPERATOR_CHARS).
# re.match anchors at position 0; \s* only allows leading whitespace, never leading prose.
_LABELLED_RE = re.compile(r"\s*answer\s*[:=]\s*(-?\d[\d,]*)", re.IGNORECASE)
# The answer label alone, with nothing after it — an empty prefill continuation, not reasoning.
_BARE_LABEL_RE = re.compile(r"answer\s*[:=]?", re.IGNORECASE)
# Otherwise the response may LEAD with a standalone integer (e.g. "1902", or "612 because …" — the
# model answered first, reasoning may follow).
_LEADING_RE = re.compile(r"\s*(-?\d[\d,]*)")


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace(" ", "")
    try:
        return int(s)
    except ValueError:
        return None


class IntegerScorer:
    """The math integer exact-match rule. No-CoT means the answer must come FIRST; we accept
    exactly two immediate forms:
      1. `Answer: N` at the start — the integer right after the label (trailing prose is fine, but
         an operator right after the integer makes it the head of an expression — CoT, not answer);
      2. the response LEADS with an integer (`612`, or `612 because…`) — same operator rule.
    A response that opens with anything else ("I need to evaluate this step by step…") has no
    immediate answer, so parse returns None rather than digging a number out of the reasoning
    (which would reward ignoring the no-CoT constraint and often grab an intermediate value).
    Scoring compares as integers when both sides are integer-valued, else exact string match."""

    @staticmethod
    def parse_answer(text: str) -> Optional[str]:
        """The model's immediate numeric answer, or None if the response doesn't lead with one."""
        if not text:
            return None
        text = _normalize_unicode(text)  # U+2212 minus, NBSP, fullwidth digits -> ASCII
        # The operator guard applies to BOTH forms: "Answer: -4 % 96 = 92 ..." is an expression —
        # chain-of-thought in the answer slot, not an answer — same as the label-less "-4 % 96 =".
        # Prefill re-attaches "Answer:" to every response, so guarding only the label-less form
        # would grade prefill-channel CoT as an immediate answer.
        m = _LABELLED_RE.match(text) or _LEADING_RE.match(text)
        if m:
            after = text[m.end():].lstrip()
            if not _is_expression_tail(after):  # a number (incl. "28%"), not the head of an expr
                return m.group(1).strip()
        return None

    @staticmethod
    def score(parsed: Optional[str], gold: Any) -> bool:
        """True if the parsed answer matches the gold answer."""
        if parsed is None:
            return False
        pi, gi = _to_int(parsed), _to_int(gold)
        if pi is not None and gi is not None:
            return pi == gi
        return str(parsed).strip() == str(gold).strip()

    @staticmethod
    def answer_form(text: str) -> str:
        """Classify HOW the model responded, for measuring no-CoT compliance per response:
          - "immediate": led with the answer (Answer: N, or a leading integer) — no-CoT respected;
          - "reasoning_first": produced output but did NOT lead with a number — no-CoT VIOLATED;
          - "empty": no usable output (including a bare "Answer:" prefill continuation)."""
        if not text or not text.strip():
            return "empty"
        if _BARE_LABEL_RE.fullmatch(text.strip()):
            return "empty"  # "Answer:" with nothing after it — an empty prefill continuation, not CoT
        return "immediate" if IntegerScorer.parse_answer(text) is not None else "reasoning_first"


# --- the n-hop string+int rule ---------------------------------------------------------------------

# A leading answer label to strip: "Answer:", "Answer =", "The answer is", optionally punctuated.
_LABEL_RE = re.compile(r"^\s*(?:the\s+answer\s+is|answer)\s*[:=]?\s*", re.IGNORECASE)
# An integer (optionally negative, thousands-separated) ANCHORED at the start of the answer text.
_LEADING_INT_RE = re.compile(r"^(-?\d[\d,]*)")
# Surrounding quotes/brackets and trailing sentence punctuation to peel off a parsed answer.
_TRIM = " \t\r\n\"'`.!,;:"


def _strip_label(text: str) -> str:
    """Remove a single leading answer label if present (else return text unchanged)."""
    return _LABEL_RE.sub("", text, count=1)


def _as_int(value: Any) -> Optional[int]:
    """Interpret a gold value as an integer: int, or an all-integer string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def _leading_int(text: str) -> Optional[int]:
    """The integer at the START of the parsed answer (the one right after "Answer:"), or None.

    Rejects a leading integer immediately followed by an operator-with-operand — the first operand
    of a chain-of-thought expression, not an immediate answer. Mirrors IntegerScorer's rule
    (shared `_is_expression_tail`, incl. its bare-trailing-operator allowance)."""
    s = text.strip()
    m = _LEADING_INT_RE.match(s)
    if not m:
        return None
    if _is_expression_tail(s[m.end():].lstrip()):
        return None
    return int(m.group(1).replace(",", ""))


def _norm(s: str) -> str:
    """Normalize a string answer for comparison: Unicode variants -> ASCII (both sides pass
    through, so the comparison stays symmetric), casefold, collapse whitespace, trim edges."""
    return re.sub(r"\s+", " ", _normalize_unicode(str(s)).strip(_TRIM)).casefold()


class StringIntScorer:
    """The n-hop fact-composition rule. Unlike the math rule (integer answers only), n-hop answers
    are MIXED type: strings (element / state / motto / flower / person names) AND integers (birth
    years, county counts). The parser isolates the answer STRING right after the "Answer:" label
    (first non-empty line, label stripped, quotes/punctuation trimmed); the scorer decides how to
    compare — numeric gold: the integer must appear at the START of that answer text; string gold:
    normalized exact match, NO fuzzy/alias matching."""

    @staticmethod
    def parse_answer(text: str) -> Optional[str]:
        """The model's immediate answer as a cleaned string, or None if there's no usable output."""
        if not text or not text.strip():
            return None
        text = _normalize_unicode(text)  # curly quotes/apostrophes, U+2212, NBSP -> ASCII
        lines = _strip_label(text.strip()).splitlines()
        cleaned = lines[0].strip(_TRIM) if lines else ""
        return cleaned or None

    @staticmethod
    def score(parsed: Optional[str], gold: Any) -> bool:
        """True iff the parsed answer matches gold. Numeric gold -> integer-right-after-Answer
        match; string gold -> normalized exact match."""
        if parsed is None:
            return False
        gi = _as_int(gold)
        if gi is not None:                      # numeric answer: integer at the start of the answer text
            return _leading_int(parsed) == gi
        return _norm(parsed) == _norm(gold)     # string answer: normalized exact match

    @staticmethod
    def answer_form(text: str) -> str:
        """Classify no-CoT compliance of a response:
          - "empty":           no usable output;
          - "immediate":       led with the answer (label or a short first line) — no-CoT respected;
          - "reasoning_first": produced prose before answering — no-CoT VIOLATED.
        Prose = two or more sentence-like segments of >= 3 words each; a period inside an
        abbreviated NAME ("St. Paul", "John F. Kennedy") splits into sub-3-word fragments and does
        not count, so multi-word answers containing periods are not misread as reasoning."""
        if not text or not text.strip():
            return "empty"
        s = _normalize_unicode(text.strip())
        if _LABEL_RE.match(s):
            return "immediate"
        first_line = s.splitlines()[0]
        segments = [seg for seg in re.split(r"\.\s+", first_line) if seg.strip()]
        is_prose = sum(len(seg.split()) >= 3 for seg in segments) >= 2
        if len(s.split()) <= 12 and len(s.splitlines()) <= 2 and not is_prose:
            return "immediate"
        return "reasoning_first"


INTEGER = IntegerScorer()
STRING_INT = StringIntScorer()


def nocot_violation(usage: Any, tool_violation: Any = None) -> bool:
    """True if a response violated the no-CoT constraint regardless of its answer: reported
    reasoning/thinking tokens OR visible reasoning content (some OpenRouter providers return the
    reasoning text while omitting the token count), or a structured-channel tool violation
    (truncated tool JSON — deliberation ate the output budget — or extra keys beside "answer").
    The single shared rule; a violating row scores wrong, is recorded, and is never excluded.

    Tolerance: reasoning_tokens <= 1 with ZERO reasoning content is NOT a violation. Some
    mandatory-reasoning endpoints bill a constant 1-token floor — an opened-and-closed think block
    — on responses that answer immediately with no reasoning text; on a live probe every immediate
    answer read exactly (1 token, 0 chars) while every deliberating response read 44-47 tokens WITH
    content, so the chars guard still catches real reasoning even at low token counts."""
    u = usage or {}
    return bool(u.get("reasoning_tokens", 0) > 1 or u.get("reasoning_chars", 0) > 0 or tool_violation)


def rescore_record(rec: dict) -> dict:
    """Recompute a stored record's parsed/answer_form/correct through the CURRENT scorers,
    exactly as the sweep driver scores at collection time (incl. the no-CoT violation rule).
    Shared by the replay regression gate."""
    from . import registry  # runtime import: registry imports this module at load time
    scorer = registry.DATASETS[rec["dataset"]].scorer
    raw = rec.get("raw_output") or ""
    parsed = scorer.parse_answer(raw)
    correct = scorer.score(parsed, rec["gold"]) and not nocot_violation(
        rec.get("usage"), rec.get("tool_violation"))
    return {"parsed": parsed, "answer_form": scorer.answer_form(raw), "correct": correct}
