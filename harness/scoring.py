"""All dataset-dependent answer parsing and scoring, in one place.

Every scorer exposes the 3-method surface the driver dispatches on (`parse_answer`, `score`,
`answer_form`) and shares the IMMEDIATE-answer stance that defines this study: the answer must
LEAD the response. A model that reasons first scores wrong even if it lands the value later —
a scorer that walked backward to the last answer-looking line would reward chain-of-thought and
defeat the no-CoT measurement, so every scorer here reads the first line only.

  - IntegerScorer     the math rule (Gen-Arithmetic / competition math): the integer after
                      "Answer:" (or leading the response), exact-match against an integer gold.
  - StringIntScorer   the n-hop fact-composition rule: mixed string/int golds; integer golds match
                      the integer anchored right after "Answer:", string golds match after the
                      normalization pass in `_norm_string` (accents, punctuation, a small set of
                      known name/place aliases, and a guarded middle-word removal for person
                      names — see that function's docstring).

This module also owns the ONE definition of the no-CoT violation rule (`nocot_violation`) and the
record re-scoring recipe (`rescore_record`) shared by the sweep driver and the replay regression
gate — a rule change lands everywhere at once or nowhere.
"""
from __future__ import annotations

import re
import unicodedata
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


# --- string-answer normalization (n-hop fact-composition golds) -----------------------------------
# Exact-match-after-normalization: no fuzzy or substring matching, just enough normalization that
# a formatting or naming-convention difference (an accent, an inserted middle name, an Oxford
# comma, a common alternate name) doesn't register as a wrong answer. Applied symmetrically to
# both the parsed answer and gold.
#
# The reference data below (person-name aliases, state mottos/flowers) is plain factual reference
# data, not code, drawn from the same fact domain the n-hop generator itself uses (that generator's
# fact tables are vendored verbatim under harness/data_curation/vendor/multi_hop/inputs/) —
# hardcoded here as plain strings since this module has no reason to import the generator.
_SUFFIXES = {"jr", "sr", "i", "ii", "iii", "iv", "v"}

# Historical figures whose commonly-used name isn't simply first+last: the generic middle-word
# rule below would otherwise pick the wrong two words (e.g. "John Boyd Orr" -> "John Orr" instead
# of the actually-used "Boyd Orr").
_NAME_ALIASES = {
    "robert bruce merrifield": "bruce merrifield",
    "oscar arias sanchez": "oscar arias",
    "john boyd orr": "boyd orr",
    "hermann emil fischer": "emil fischer",
    "petrus debye": "peter debye",
    "adolf otto reinhold windaus": "adolf windaus",
    "william randal cremer": "randal cremer",
    "rigoberta menchu tum": "rigoberta menchu",
    "sharlene wells hawkes": "sharlene wells",
}
# First-name nicknames, applied as a word-boundary substitution rather than a fixed-phrase alias
# above: a fixed "judith ford" -> "judi ford" entry would have diverted GOLD away from "judith
# ford" while a longer parsed form like "Judith Anne Ford" still reduces to "judith ford" via the
# middle-word rule below, breaking a match that already worked. Substituting the nickname itself
# lets every form (Judi Ford / Judith Ford / Judith Anne Ford) converge on the same string.
_NICKNAME_RE = re.compile(r"\bjudi\b")
# American/British spelling variants for chemical elements. Gold values in this dataset are
# always the American spelling; these map an alternate spelling down to it.
_ELEMENT_ALIASES = {
    "aluminium": "aluminum",
    "caesium": "cesium",
}
# US state mottos/flowers (all 50 states + DC) — non-name multi-word phrases that the middle-word
# remover below must NOT touch (it would mangle e.g. Colorado's 3-word Latin motto "Nil Sine
# Numine" by treating it as a person's name and dropping the middle word).
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
    "yucca flower": "yucca",
    "wild native sunflower": "sunflower",
    "wild sunflower": "sunflower",
    "texas bluebonnet": "bluebonnet",
    "scarlet carnation": "carnation",
    "wyoming indian paintbrush": "indian paintbrush",
    "delaware peach blossom": "peach blossom",
    # NOT reduced to bare "rhododendron": that's a SEPARATE state's actual flower name in this
    # same list (West Virginia), so collapsing "Coast Rhododendron" (Washington) down to it would
    # make the two states' answers indistinguishable.
    "washington rhododendron": "coast rhododendron",
    # "common meadow violet": "violet" deliberately OMITTED (unlike the rest of this dict, which
    # matches the ones in the paper's own normalize_answer) — New Jersey's "Common Meadow
    # Violet" would collapse onto Illinois/Rhode Island's own actual "Violet", making three
    # different states' gold answers indistinguishable. Confirmed via a full collision check
    # across every string-valued fact table our vendored generator uses.
}
# "american " INCLUDED despite colliding with a different state's gold answer (Virginia's
# "American Dogwood" -> "dogwood", same normalized form as North Carolina's "Dogwood"): the two
# states' state flower really is the same species (Flowering Dogwood, Cornus florida) in real
# life, so crediting the match is a deliberate call, not an oversight — confirmed with the
# project owner 2026-07-28 after the collision check first flagged it.
_DIRECTIONAL_PREFIXES = ("northern ", "western ", "eastern ", "southern ", "american ")
_HONORIFIC_PREFIXES = ("mr. ", "sir. ", "mr ", "sir ", "lord ")


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _remove_middle_word(s: str) -> str:
    """First + last word for a 3-4 word all-alphabetic phrase (a middle name/initial), preserving
    a trailing suffix (Jr/Sr/I-V). A 4-word phrase WITHOUT a suffix is left unchanged (only a
    single inserted word is stripped, never two)."""
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


def _base_normalize_string(s: str) -> str:
    """Every normalization step except the final middle-word removal — factored out so
    `_NON_NAME_SET` (built below) can be compared against pre-middle-word-removal strings."""
    s = _strip_accents(_normalize_unicode(s).strip()).casefold()
    for prefix in _HONORIFIC_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r"\s*\([^)]*\)", "", s)  # parenthetical explanation, e.g. "X (also known as Y)"
    # A trailing " - <clause>" / " — <clause>" is an appended gloss ("<answer> - West Virginia"),
    # not part of the answer itself. Requires whitespace on both sides of the dash so a
    # hyphenated compound word ("Al-ki", "Forget-me-not") is never affected.
    s = re.sub(r"\s+[-–—]\s+.*$", "", s)
    s = _NICKNAME_RE.sub("judith", s)
    for alias, canon in _NAME_ALIASES.items():
        s = s.replace(alias, canon)
    for alias, canon in _ELEMENT_ALIASES.items():
        s = s.replace(alias, canon)
    # Washington DC and Alabama's motto each have two equally-valid forms in general use (place
    # name vs "District of Columbia"; English translation vs the original Latin) — canonicalize
    # both directions so either form matches the other.
    s = s.replace("washington, d.c.", "district of columbia")
    s = s.replace("d.c.", "district of columbia")
    if s == "washington":
        s = "district of columbia"
    s = s.replace("audemus jura nostra defendere", "we dare defend our rights")
    s = s.replace("fatti maschi,", "fatti maschii,")  # a common misspelling of Maryland's motto
    for ch in ",.-'`":
        s = s.replace(ch, "")
    s = re.sub(r"\band\b", "", s)  # drop the word "and" -> Oxford-comma-insensitive
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    for prefix in _DIRECTIONAL_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    for alias, canon in _FLOWER_ALIASES.items():
        s = s.replace(alias, canon)
    return s.strip()


_NON_NAME_SET = {_base_normalize_string(x) for x in _STATE_MOTTOS + _STATE_FLOWERS}


def _norm_string(s: str) -> str:
    """Normalize a string answer for comparison — Unicode/accent folding, casefold, punctuation
    and Oxford-comma insensitivity, a small set of historical-figure aliases, and a narrowly
    scoped middle-word remover for person names — guarded against state mottos/flowers, which
    are non-name multi-word phrases the middle-word rule must not touch."""
    base = _base_normalize_string(s)
    if base in _NON_NAME_SET:
        return base
    return _remove_middle_word(base)


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
        return _norm_string(parsed) == _norm_string(gold)  # string answer: normalized exact match

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


# Some mandatory-reasoning endpoints bill a constant 1-token floor — an opened-and-closed think
# block — on responses that answer immediately with no reasoning text; on a live probe every
# immediate answer read exactly (1 token, 0 chars) while every deliberating response read 44-47
# tokens WITH content, so >1 (not >0) is the real threshold and the chars guard still catches
# real reasoning even at low token counts. Shared by `nocot_violation` and `classify_response` so
# the two can never disagree about where "reasoning happened" starts.
_REASONING_TOKEN_FLOOR = 1


def nocot_violation(usage: Any, tool_violation: Any = None) -> bool:
    """True if a response violated the no-CoT constraint regardless of its answer: reported
    reasoning/thinking tokens OR visible reasoning content (some OpenRouter providers return the
    reasoning text while omitting the token count), or a structured-channel tool violation
    (truncated tool JSON — deliberation ate the output budget — or extra keys beside "answer").
    The single shared rule; a violating row scores wrong, is recorded, and is never excluded."""
    u = usage or {}
    return bool(u.get("reasoning_tokens", 0) > _REASONING_TOKEN_FLOOR
                or u.get("reasoning_chars", 0) > 0 or tool_violation)


def classify_response(rec: dict) -> str:
    """Label one stored record by WHAT KIND of no-CoT behavior it shows — distinct from
    `nocot_violation`'s single yes/no, since "the model refused" and "the model reasoned but
    hid it" call for different follow-up (a refusal is a channel/prompt problem; hidden reasoning
    is a measurement problem). One of:

      "error"              — the call itself failed (transport/API error, not a real observation).
      "external_thinking"  — reasoning is VISIBLE: prose before the answer (`answer_form ==
                              "reasoning_first"`), reasoning content in a provider's dedicated
                              field (`reasoning_chars > 0`), or a structured tool call that
                              smuggled an extra key or got truncated by deliberation.
      "internal_thinking"  — reasoning is HIDDEN: a reported token count with no visible text
                              (`reasoning_tokens` above the 1-token reporting floor) — what the
                              structured-channel token-cost comparison is built to catch, since
                              `reasoning_tokens == 0` alone can't distinguish "didn't reason" from
                              "reasoned but the API has no field to report it in."
      "refusal"             — no usable output AND no reasoning signal of either kind: the model
                              declined to engage, not a case of unreported computation.
      "clean"               — an immediate answer with neither signal — genuine no-CoT compliance.

    External is checked before internal: a response can show both signals, and visible reasoning
    is the stronger, more specific claim (internal-only is inferred from a token count alone).
    """
    if rec.get("error") is not None:
        return "error"
    u = rec.get("usage") or {}
    tool_violation = rec.get("tool_violation")
    if (rec.get("answer_form") == "reasoning_first" or u.get("reasoning_chars", 0) > 0
            or tool_violation in ("extra_keys", "truncated")):
        return "external_thinking"
    if u.get("reasoning_tokens", 0) > _REASONING_TOKEN_FLOOR:
        return "internal_thinking"
    if rec.get("answer_form") == "empty":
        return "refusal"
    return "clean"


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
