"""Model backends — how each model is forced to answer with NO chain-of-thought (a single
forward pass), all reached via OpenRouter's alias namespace (see the root README's transport
note for why every model goes through OpenRouter rather than a native provider SDK).

  OpenAICompatBackend  (base class: the OpenAI-compatible chat-completions request shape)
    - append "Answer:" to the final user message, so the model's immediate continuation is the
      no-CoT answer (the shape every OpenRouter backend below builds on).

  OpenRouterBackend  (append — most models resolve here)
    - reasoning disabled via OpenRouter's UNIFIED control, extra_body={"reasoning": {"enabled":
      False}}, which turns thinking off on any reasoning-capable model and is ignored by models
      that don't reason.
    - reported reasoning_tokens > 0 is scored WRONG regardless (`scoring.nocot_violation`).
      CAVEAT: some OpenRouter providers omit the reasoning-token count, so a zero is necessary
      but not sufficient — the reasoning-disable is what actually enforces no-CoT.

  OpenRouterAdaptiveBackend  (append, for models that reject the reasoning-disable outright)
    - some endpoints 400 on the unified disable ("Reasoning is mandatory ...") — verified live
      for the adaptive-thinking-only model this repo studies. Sends NO reasoning param at all and
      relies entirely on a strict system prompt + scoring any reported reasoning as wrong.

  OpenRouterPrefillBackend  (prefill — Claude ids that honor a trailing assistant continuation)
    - OpenRouter forwards a trailing assistant message to the pinned Anthropic provider as an
      assistant prefill, so the model continues an assistant turn that already says "Answer:".

  OpenRouterStructuredBackend / OpenRouterToolBackend  (structured)
    - the answer is returned as STRUCTURED data (a JSON schema response, or a forced tool call
      for Claude ids, which OpenRouter translates into Anthropic's native tool-call format) —
      no free-text chain-of-thought is possible in the output. This is the adaptive-only model's
      default channel here (see the root README): forcing the output shape makes free text
      impossible regardless of whether a reasoning-disable parameter would have worked, though
      that is not, by itself, proof that no internal reasoning pass occurred.

NOTE ON FEW-SHOT FORMAT: the demonstrations are identical regardless of channel — each shows the
assistant answering "Answer: N". Only the final (query) turn differs: prefill continues an
assistant turn, append/structured elicit from the user turn — unless condition-matched
(`prompt.build_messages`'s `demo_cond`), which renders every demo through the query's condition
too, trading the plain-demo cacheable prefix for a stricter apples-to-apples comparison.

TRANSPORT: every backend above reaches its model via OpenRouter. The `AnthropicNative*` backends
below are an alternate TRANSPORT for "anthropic/..." ids only — same methods (prefill/append/
structured-as-tool), same message assembly, but hitting api.anthropic.com directly instead. Each
class's own docstring covers its no-CoT mechanics, same as the OpenRouter backends above.
  - `wants_cache_control = True` here: unlike OpenRouter (found live to never cache these models
    regardless of prompt length), native Anthropic honors an explicit `cache_control` breakpoint,
    so the few-shot prefix can genuinely cache on this transport.
  - Selected via `transport="anthropic_native"` on `backend_for`/`request_params` (default stays
    `"openrouter"` — nothing above changes). A separate async path, `harness/anthropic_batch.py`,
    reuses these same backends' `build_params` to submit the identical request shape through the
    Message Batches API instead of a live call, at its 50% discount.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from . import conditions, prompt
from .prompt import PREFILL

# Token budget per answer call. The experiment's extra compute comes from the INPUT
# (repeats/filler); output beyond the short answer would itself be reasoning. Free-text answers
# use well under this; the structured tool-call/JSON scaffolding needs the higher
# STRUCTURED_MIN_TOKENS floor below.
MAX_ANSWER_TOKENS = 50

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Pin OpenRouter requests to ONE backing serving stack per model family. OpenRouter otherwise
# load-balances a namespaced id across backing providers per request, which would silently mix
# serving stacks inside one experiment — and a no-CoT compliance verdict measured on one stack
# does not describe another. Namespace prefix -> provider slug; a namespace absent here keeps
# OpenRouter's default routing.
#   anthropic/  -> Anthropic's own stack (verified live: responses report provider=="Anthropic").
#   moonshotai/ -> Moonshot's own stack. kimi-k3's clean-compliance evidence (2026-08-05) was
#                  collected across THREE stacks (Chutes/DigitalOcean/Fireworks), so a pin is
#                  required before any real run; the first-party endpoint is the natural choice
#                  and prices the same as the default ($3/$15 per M, checked live).
#   google/     -> Google AI Studio. Both available stacks are Google first-party (AI Studio
#                  and Vertex) at identical prices; the 2026-08-05/07 probe evidence mixed the
#                  two, so one is fixed here — AI Studio, the surface that served the clean
#                  probe calls for every elicitable Gemini.
_PROVIDER_PINS = {"anthropic/": "anthropic", "moonshotai/": "moonshotai",
                  "google/": "google-ai-studio",
                  # Open-weight arm (2026-08-11, from the nocot tool's probed profiles):
                  # author's own stack where one exists, else the best full-precision host.
                  "qwen/": "alibaba", "minimax/": "minimax/fp8", "z-ai/": "z-ai",
                  "nvidia/": "together", "meta-llama/": "google-vertex/us-east5",
                  # gpt-6-astra (2026-09-09): pinned by EXACT id, not the "openai/" namespace —
                  # gpt-5.6-sol's stored rows were collected unpinned and must keep that request
                  # shape. OpenRouter lists three OpenAI first-party tags for Astra plus Azure;
                  # "openai" is the standard tier ($10/$50). Switch to "openai/flex" (half price,
                  # lower priority, same weights) only as a deliberate cost decision.
                  "openai/gpt-6-astra": "openai"}

# Models whose endpoint rejects OpenRouter's unified reasoning-disable: no disable is ever sent
# — they route to OpenRouterAdaptiveBackend under `append`, and skip the disable param under
# `structured` and `prefill` too (enforcement falls to the channel shape + the strict prompt +
# `scoring.nocot_violation`). fable-5 verified live to 400 on the disable ("Reasoning is
# mandatory for this endpoint and cannot be disabled"). opus-5 remains listed on THIS
# (OpenRouter) side even though the endpoint now accepts the disable on both transports (probed
# 2026-08-04, see `_NATIVE_EXPLICIT_DISABLE`): every stored OpenRouter opus-5 row was collected
# without the disable, so flipping this entry would silently change what its OpenRouter channels
# send — a separate decision to make explicitly, with its own re-run, not as a side effect.
# The Gemini rows are from the 2026-08-05/07 probe cards: every listed endpoint 400s the
# disable the same way ("Reasoning is mandatory"); listing them here converts those hard 400s
# into runnable adaptive-regime channels.
_MANDATORY_REASONING = {
    "anthropic/claude-fable-5", "anthropic/claude-opus-5",
    "google/gemini-2.5-pro", "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash", "google/gemini-3.5-flash-lite", "google/gemini-3.6-flash",
    # gpt-6-astra (probed 2026-09-09): the pinned
    # OpenAI endpoint 400s the unified disable with the same "Reasoning is mandatory" text —
    # unlike gpt-5.6-sol, which accepts it. Its forced-tool probe calls billed 0 reasoning
    # tokens; its prefill probe calls billed a fixed 11-12 (scored as violations).
    "openai/gpt-6-astra",
    # fable-5.1 (probed 2026-09-10, both transports): unified disable 400s ("Reasoning is
    # mandatory"), native thinking.type.disabled/enabled both 400 (adaptive + effort low is the
    # floor, "none" invalid), assistant prefill hard-400s, and — unlike fable-5 — tool_choice
    # type "tool"/"any" is NOT supported, so the forced-tool structured channel is gone.
    "anthropic/claude-fable-5.1",
}

# Models verified live (probe cards: 3/3 numeric continuation, 0 reasoning) to honor a trailing
# assistant message as a genuine prefill continuation via OpenRouter. Not Claude-only: most
# Gemini endpoints forward the shape cleanly too — for the mandatory-reasoning ones this is
# their ONLY probe-clean channel. (gpt-5.6-sol is deliberately absent: it answers as a fresh
# turn instead of continuing — "silently_dropped". gemini-3.6-flash and 3.5-flash-lite hard-400
# the shape itself.)
_SUPPORTS_PREFILL = {
    "anthropic/claude-opus-4.5",
    # gpt-6-astra (probe 2026-09-09): the trailing assistant turn IS forwarded as a genuine
    # continuation (3/3 numeric continuation, no restated marker) — the first OpenAI id to do
    # so (sol "silently_dropped"). Listed so the channel can be MATRIXED; the probe calls
    # billed 11-12 reasoning tokens each, so it is not clean until the strict-prompt variant
    # proves otherwise on real data.
    "openai/gpt-6-astra",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-pro",
    "google/gemini-3.5-flash",
}

# Closed-think prefill is gated separately from plain prefill: an endpoint that forwards a
# trailing assistant message does NOT necessarily treat a closed think block as a completed
# reasoning phase (the nocot suite saw the same text go 32/32 violations on llama-4). Only
# models whose closed-think style is probe-verified clean belong here.
_SUPPORTS_CLOSED_PREFILL = {"google/gemini-2.5-pro"}


def _or_extra_body(model: str, *, reasoning_off: bool = True) -> Dict[str, Any]:
    """The OpenRouter extra_body for `model`: the unified reasoning-disable (skipped for models
    that reject it) plus this namespace's provider pin, if it has one (`_PROVIDER_PINS`). EVERY
    OpenRouter request path builds its extra_body here, so a new backend cannot forget the pin.
    May return {} — callers that must not send an empty extra_body should only attach it when
    truthy."""
    body: Dict[str, Any] = {"reasoning": {"enabled": False}} if reasoning_off else {}
    for prefix, provider in _PROVIDER_PINS.items():
        if model.startswith(prefix):
            body["provider"] = {"only": [provider], "allow_fallbacks": False}
            break
    return body


class OpenAICompatBackend:
    """The OpenAI-compatible chat-completions request shape: no-CoT by appending "Answer:" to the
    final user message. Shared base for every OpenRouter backend below — never used directly
    (OpenRouterBackend overrides `client()` with OpenRouter's base_url)."""

    method = "append"
    wants_cache_control = False      # OpenAI-compatible caching is automatic; no markers needed.
    system_base: str | None = None   # None = use the dataset's instruction (no strict variant here).
    no_cot_enforcement = "append 'Answer:' to the user message"

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        msgs = [{"role": "system", "content": system}] + list(messages)
        tail = msgs[-1]
        msgs[-1] = {"role": "user", "content": f"{tail['content']}\n{PREFILL}"}
        return {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}

    def extract_text(self, resp: Any) -> str:
        return resp.choices[0].message.content or ""

    def usage_dict(self, resp: Any) -> Dict[str, int]:
        u = resp.usage
        cdetails = getattr(u, "completion_tokens_details", None)
        reasoning = (getattr(cdetails, "reasoning_tokens", 0) or 0) if cdetails else 0
        return {
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
            "reasoning_tokens": reasoning,
        }

    def complete(self, client: Any, model: str, messages: List[Dict[str, Any]], *,
                 system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Tuple[Any, str]:
        # Single sample, recorded as-is. A response that reports reasoning tokens despite the
        # disable is recorded and scored wrong, never dropped.
        params = self.build_params(model, messages, system=system, max_tokens=max_tokens)
        resp = client.chat.completions.create(**params)
        # OpenRouter (and some compat gateways) relay provider errors IN-BODY with HTTP 200 and
        # choices=None — e.g. {"error": {"message": "Overloaded", "code": 503}} — which the SDK
        # does not raise. Surface it as the row's error (clean, retryable on resume) instead of
        # letting extract_text crash with an opaque NoneType TypeError.
        if getattr(resp, "choices", None) in (None, []):
            err = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
            raise RuntimeError(f"provider returned no choices: {err or 'unknown error'}")
        return resp, self.extract_text(resp)


class OpenRouterBackend(OpenAICompatBackend):
    """OpenRouter gateway — reaches any model by its namespaced id, e.g. "openai/gpt-5.5".

    No-CoT: reasoning disabled via OpenRouter's UNIFIED control extra_body={"reasoning":
    {"enabled": False}} + the inherited "Answer:" append; reported reasoning_tokens>0 is scored
    wrong regardless."""

    method = "append_noreason"
    no_cot_enforcement = ("reasoning {enabled:false} (OpenRouter unified) + append 'Answer:'; "
                          "reported reasoning_tokens>0 scored wrong")

    def client(self) -> Any:
        import httpx
        import openai
        # Real timeouts (2026-09-10): during the astra collection a fraction of OpenRouter
        # requests stalled on open connections until the SDK's default 600s timeout, then
        # retried and succeeded — no error rows, just a throughput collapse from ~700k to
        # ~360k tokens/min while six parallel probe requests from a fresh client returned in
        # 3s. A 60s read timeout fails a stalled request fast; the SDK retries it (3 tries)
        # and a still-failing cell lands as an error row, retried on resume. Request
        # content is unchanged — this is transport plumbing only.
        return openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={"X-Title": "no-cot-harness"},
            timeout=httpx.Timeout(60.0, connect=10.0),
            max_retries=3,
        )

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        params = super().build_params(model, messages, system=system, max_tokens=max_tokens)
        params["extra_body"] = _or_extra_body(model)
        return params

    def usage_dict(self, resp: Any) -> Dict[str, int]:
        # Beyond token counts: OpenRouter's UNIFIED interface can return the reasoning CONTENT
        # itself (`message.reasoning`) even when a provider omits the reasoning-token count —
        # visible chain-of-thought the token-based rule alone would miss.
        u = super().usage_dict(resp)
        msg = resp.choices[0].message
        reasoning = (getattr(msg, "reasoning", None)
                     or (getattr(msg, "model_extra", None) or {}).get("reasoning") or "")
        u["reasoning_chars"] = len(reasoning)
        return u


class OpenRouterAdaptiveBackend(OpenRouterBackend):
    """OpenRouter endpoints where reasoning is MANDATORY and cannot be disabled — the unified
    control 400s ("Reasoning is mandatory for this endpoint and cannot be disabled"). Sends NO
    reasoning param at all (skip OpenRouterBackend's disable) and lets the inherited usage_dict's
    reasoning_tokens/reasoning_chars feed the universal `scoring.nocot_violation` rule — any
    nonzero reading is scored wrong, never suppressed. The Anthropic provider PIN still applies
    to anthropic/* ids: dropping the reasoning param must not also drop the transport policy.

    system_base is a strict no-think prompt used as a functional default for these models, since
    the API parameter can't do the job."""

    method = "openrouter_adaptive"
    system_base = prompt.STRICT_NOCOT_PROMPT
    no_cot_enforcement = ("strict no-think prompt (reasoning cannot be disabled on this endpoint); "
                          "any reasoning_tokens>0 or reasoning content scored wrong; provider "
                          "pinned to Anthropic for anthropic/* ids")

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        params = OpenAICompatBackend.build_params(self, model, messages, system=system,
                                                  max_tokens=max_tokens)
        extra = _or_extra_body(model, reasoning_off=False)
        if extra:
            params["extra_body"] = extra
        return params


class OpenRouterPrefillBackend(OpenRouterBackend):
    """Prefill via OpenRouter: a trailing assistant message is forwarded to the backing provider
    as an assistant prefill, so the model continues an assistant turn that already says
    "Answer:" (verified live: continuation arrives without the prefix, reasoning_tokens=0).
    `backend_for` gates the route on `_SUPPORTS_PREFILL` — Claude ids and the Gemini endpoints
    whose probes show a genuine continuation; for mandatory-reasoning models this structural
    channel is the only probe-clean one, and no disable param is sent (see build_params)."""

    method = "openrouter_prefill"
    no_cot_enforcement = ("prefill 'Answer:' (trailing assistant message via OpenRouter) + "
                          "reasoning disabled where the endpoint accepts it (strict prompt "
                          "where it doesn't) + provider pin where configured + small max_tokens")

    def __init__(self, model: str = "") -> None:
        # Mandatory-reasoning models get the strict prompt on this channel too: the 2026-08-07
        # elicitation suite found prefill:strict fully clean on gemini-3.1-pro (16/16) while
        # prefill:plain let one baseline call reason — same belt-and-suspenders as the adaptive
        # append channel. Disable-accepting models keep the dataset's plain prompt.
        if model in _MANDATORY_REASONING:
            self.system_base = prompt.STRICT_NOCOT_PROMPT

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        msgs = ([{"role": "system", "content": system}] + list(messages)
                + [{"role": "assistant", "content": PREFILL}])
        # Mandatory-reasoning endpoints 400 on the disable param, so they get the pure prefill
        # shape (same fallback as the structured backends) — the continuation structure is the
        # enforcement, and any reasoning still scores wrong via `scoring.nocot_violation`.
        return {"model": model, "messages": msgs, "max_completion_tokens": max_tokens,
                "extra_body": _or_extra_body(model,
                                             reasoning_off=model not in _MANDATORY_REASONING)}

    def extract_text(self, resp: Any) -> str:
        return PREFILL + (resp.choices[0].message.content or "")


class OpenRouterClosedThinkPrefillBackend(OpenRouterPrefillBackend):
    """Closed-think prefill: the trailing assistant message opens AND CLOSES an empty think
    block before the "Answer:" cue, so an always-thinking endpoint that keys its mandatory
    reasoning pass off the reply structure treats the phase as already completed and continues
    straight into the answer. The block is ours — the model never generates a token inside it.
    Channel text mirrors the nocot elicitation tool's probe-gated style byte-for-byte; that
    tool's evidence for gemini-2.5-pro: 0 violations, all answers immediate, on every probed
    cell (gen_arithmetic baseline/f300/f1000, 8/8 each; comp_math and nhop_2 piloted to f100),
    where the OPEN prefill — plain and strict variants both — lets the thinking pass re-fire
    on gen f300/f1000 rows (27-42 billed reasoning tokens mid-continuation). observed_only
    enforcement: an implementation
    behavior, not a contract — re-pilot per run. Same strict-prompt policy as the parent for
    mandatory-reasoning models; any leak still scores wrong via `scoring.nocot_violation`."""

    method = "openrouter_prefill_closed"
    _CLOSED_PREFILL = f"<think>\n</think>\n\n{PREFILL}"
    no_cot_enforcement = ("prefill closed think block + 'Answer:' (trailing assistant message) "
                          "+ strict prompt on mandatory-reasoning endpoints + provider pin + "
                          "small max_tokens")

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        params = super().build_params(model, messages, system=system, max_tokens=max_tokens)
        params["messages"][-1] = {"role": "assistant", "content": self._CLOSED_PREFILL}
        return params


# ---------------------------------------------------------------------------------------------------
# Structured-output elicitation: the answer is returned as STRUCTURED data, so no free-text
# chain-of-thought is possible in the OUTPUT — the fallback for (model, dataset) combinations that
# won't comply under prefill/append, and the adaptive-only model's default channel here.

# Output floor for the structured channel only: the tool-call/JSON scaffolding plus a long answer
# needs more room than a free-text answer, and a binding cap silently truncates the JSON before the
# answer field, which scores as a violation (below) rather than a clean answer. 100 clears every
# clean {"answer": ...} with margin; a response that still hits it is emitting deliberation.
STRUCTURED_MIN_TOKENS = 100

# The only key a structured answer may carry. Anything else ({"reasoning": ...}) is chain-of-thought
# smuggled into the answer channel; schemas forbid it and `tool_violation` reports it if a model
# emits it anyway (the API does not hard-validate tool input against the schema).
_ANSWER_ONLY_KEYS = {"answer"}

# The forced tool-call schema, keyed by the DATASET's answer type: {"answer": int} for math,
# {"answer": str} for e.g. n-hop names/phrases.
_TOOLS = {
    "integer": {
        "name": "submit_answer",
        "description": "Submit the final numerical answer to the math problem.",
        "input_schema": {"type": "object",
                         "properties": {"answer": {"type": "integer", "description": "the numerical answer"}},
                         "required": ["answer"], "additionalProperties": False},
    },
    "string": {
        "name": "submit_answer",
        "description": "Submit the single final answer to the question.",
        "input_schema": {"type": "object",
                         "properties": {"answer": {"type": "string",
                                                   "description": "the final answer: a name, word, or number"}},
                         "required": ["answer"], "additionalProperties": False},
    },
}

# JSON schemas for the response_format path (non-Claude ids), same answer-type keying as _TOOLS.
_ANSWER_JSON_SCHEMAS = {
    "integer": {
        "type": "json_schema",
        "json_schema": {"name": "math_answer", "strict": True,
                        "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                                   "required": ["answer"], "additionalProperties": False}},
    },
    "string": {
        "type": "json_schema",
        "json_schema": {"name": "final_answer", "strict": True,
                        "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                                   "required": ["answer"], "additionalProperties": False}},
    },
}


def _dict_tool_violation(tool_input, hit_max_tokens: bool):
    """Shared violation rule: "truncated" (output cap hit — deliberation ate the budget),
    "extra_keys" (reasoning smuggled beside the answer), or None (clean)."""
    if hit_max_tokens:
        return "truncated"
    if isinstance(tool_input, dict) and set(tool_input) - _ANSWER_ONLY_KEYS:
        return "extra_keys"
    return None


def _extract_json_answer(resp: Any) -> str:
    content = resp.choices[0].message.content or ""
    try:
        return str(json.loads(content).get("answer", ""))
    except (json.JSONDecodeError, AttributeError):
        return content


def _extract_json_tool_input(resp: Any) -> Any:
    try:
        obj = json.loads(resp.choices[0].message.content or "")
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _json_tool_violation(resp: Any):
    return _dict_tool_violation(_extract_json_tool_input(resp),
                                getattr(resp.choices[0], "finish_reason", None) == "length")


class OpenRouterStructuredBackend(OpenRouterBackend):
    """OpenRouter structured output for non-Claude ids: response_format json-schema (strict), so
    the reply is exactly {"answer": <dataset's type>} — no free text. Mandatory-reasoning
    endpoints run without the disable param (same fallback as `append`); any reasoning is still
    caught by `scoring.nocot_violation`."""

    method = "structured_json"
    no_cot_enforcement = ("response_format json_schema (strict) + reasoning disabled where the "
                          "endpoint allows it (violations scored wrong regardless) — {\"answer\": ...}")

    def __init__(self, answer_schema: str = "integer"):
        self._response_format = _ANSWER_JSON_SCHEMAS[answer_schema]

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        msgs = [{"role": "system", "content": system}] + list(messages)
        return {"model": model, "messages": msgs,
                "max_completion_tokens": max(max_tokens, STRUCTURED_MIN_TOKENS),
                "response_format": self._response_format,
                "extra_body": _or_extra_body(model, reasoning_off=model not in _MANDATORY_REASONING)}

    def extract_text(self, resp: Any) -> str:
        return _extract_json_answer(resp)

    extract_tool_input = staticmethod(_extract_json_tool_input)
    tool_violation = staticmethod(_json_tool_violation)


class OpenRouterToolBackend(OpenRouterBackend):
    """Claude via OpenRouter under a forced tool call: the model must answer through the
    `submit_answer` tool (tool_choice forces it — OpenRouter translates the OpenAI-style function
    spec into Anthropic's native tool format), so it cannot emit free-text reasoning in the
    output. The adaptive-only model's default channel here (see the root README) — `tool_choice`
    forces WHICH tool is called, not that no reasoning preceded it, so treat this channel's
    numbers with that caveat in mind."""

    method = "openrouter_tool"
    no_cot_enforcement = ("forced tool use (submit_answer) via OpenRouter + reasoning disabled + "
                          "provider pinned to Anthropic — answer as a tool call, no free text")

    def __init__(self, answer_schema: str = "integer"):
        self._tool = _TOOLS[answer_schema]

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        msgs = [{"role": "system", "content": system}] + list(messages)
        return {"model": model, "messages": msgs,
                "max_completion_tokens": max(max_tokens, STRUCTURED_MIN_TOKENS),
                "tools": [{"type": "function",
                           "function": {"name": self._tool["name"],
                                        "description": self._tool["description"],
                                        "parameters": self._tool["input_schema"]}}],
                "tool_choice": {"type": "function", "function": {"name": self._tool["name"]}},
                # Same mandatory-reasoning fallback as OpenRouterStructuredBackend.
                "extra_body": _or_extra_body(model, reasoning_off=model not in _MANDATORY_REASONING)}

    def extract_tool_input(self, resp: Any) -> Any:
        for call in resp.choices[0].message.tool_calls or []:
            if call.function.name == self._tool["name"]:
                try:
                    obj = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
        return None

    def extract_text(self, resp: Any) -> str:
        obj = self.extract_tool_input(resp)
        return "" if not obj else str(obj.get("answer", ""))

    def tool_violation(self, resp: Any):
        return _dict_tool_violation(self.extract_tool_input(resp),
                                    getattr(resp.choices[0], "finish_reason", None) == "length")


# ---------------------------------------------------------------------------------------------------
# Native Anthropic transport (api.anthropic.com directly, no OpenRouter). "anthropic/..." ids only.
# NOT YET LIVE-VERIFIED (native key over its usage limit as of this writing) — a design, not a
# confirmed claim, until run against the real API.

# Opus 4.5 (extended-thinking-only): thinking is opt-in, omitting it is the no-CoT state.
#
# Fable 5 (not a general "adaptive models" rule — each model needs its own check, same as
# `_MANDATORY_REASONING`): 400s on `thinking: {type: "enabled"}` and on an explicit disable.
# Suppression attempt: `thinking: {type: "adaptive"}, output_config: {effort: "low"}` — "none" is
# NOT a valid effort value (verified live: 400s, "Input should be 'low', 'medium', 'high', 'xhigh'
# or 'max'"; "low" is the actual minimum) — + the strict prompt; `scoring.nocot_violation` is the
# real enforcement regardless of whether this helps.
_NATIVE_ADAPTIVE_THINKING = {"anthropic/claude-fable-5", "anthropic/claude-fable-5.1"}

# Opus 5 was originally in the adaptive-only set (July 2026: 400 on explicit thinking params,
# native API), but the endpoint CHANGED: re-probed 2026-08-04 (native transport,
# native transport), `thinking: {type:
# "disabled"}` is now accepted and OBEYED — 5/5 calls, 0 reasoning tokens/chars, all correct.
# So it runs with the explicit disable (API-enforced, like opus-4.5's omission — no strict
# prompt needed). The OpenRouter-side `_MANDATORY_REASONING` entry is deliberately unchanged
# in this commit: its channels are a separate decision with their own stored evidence.
_NATIVE_EXPLICIT_DISABLE = {"anthropic/claude-opus-5"}


def _native_thinking_params(model: str) -> Dict[str, Any]:
    if model in _NATIVE_EXPLICIT_DISABLE:
        return {"thinking": {"type": "disabled"}}
    if model in _NATIVE_ADAPTIVE_THINKING:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    return {}


def _native_model_id(model: str) -> str:
    """The OpenRouter alias (e.g. "anthropic/claude-opus-4.5") stays the canonical id everywhere
    in the harness (cells, signatures, the registry) — this is the one place it's translated to
    the native API's bare slug (verified live: native 404s on the OpenRouter form; wants
    "claude-opus-4-5", dots replaced with dashes)."""
    return model.removeprefix("anthropic/").replace(".", "-")


class AnthropicNativeBackend:
    """Shared native-transport plumbing: the `anthropic` SDK client, response/usage extraction for
    Anthropic's own shape (`.content` typed blocks, not `.choices[0].message`), and real caching
    (`wants_cache_control = True` — OpenRouter never caches these models; native does). Never used
    directly — subclassed per method below."""

    wants_cache_control = True
    system_base: str | None = None

    def client(self) -> Any:
        import anthropic
        # This environment carries several Anthropic keys; the xlab research workspace is
        # this project's billing home. Prefer it explicitly, fall back to the SDK default.
        return anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_XLAB_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    def extract_text(self, resp: Any) -> str:
        return next((b.text for b in resp.content if b.type == "text"), "")

    def usage_dict(self, resp: Any) -> Dict[str, int]:
        u = resp.usage
        details = getattr(u, "output_tokens_details", None)
        reasoning_tokens = (getattr(details, "thinking_tokens", 0) or 0) if details else 0
        reasoning_chars = sum(len(b.thinking) for b in resp.content if b.type == "thinking")
        return {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "reasoning_tokens": reasoning_tokens,
            "reasoning_chars": reasoning_chars,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    def complete(self, client: Any, model: str, messages: List[Dict[str, Any]], *,
                 system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Tuple[Any, str]:
        params = self.build_params(model, messages, system=system, max_tokens=max_tokens)
        resp = client.messages.create(**params)
        return resp, self.extract_text(resp)


class AnthropicNativePrefillBackend(AnthropicNativeBackend):
    """Native prefill: a trailing assistant message is a genuine continuation on Anthropic's own
    API. Only for ids in `_SUPPORTS_PREFILL` (`backend_for` gates the route) — never adaptive-only
    ids, so no thinking-suppression branch is needed here."""

    method = "anthropic_native_prefill"
    no_cot_enforcement = "prefill 'Answer:' (native trailing assistant message) + thinking omitted"

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        msgs = list(messages) + [{"role": "assistant", "content": PREFILL}]
        return {"model": _native_model_id(model), "system": system, "messages": msgs,
                "max_tokens": max_tokens}

    def extract_text(self, resp: Any) -> str:
        return PREFILL + super().extract_text(resp)


class AnthropicNativeAppendBackend(AnthropicNativeBackend):
    """Native append: "Answer:" appended to the final user turn. Adaptive-only ids get the strict
    prompt (mirrors `OpenRouterAdaptiveBackend`); since one instance serves both model kinds here
    (unlike OpenRouter's separate class per group), `system_base` is set per-model in `__init__`
    rather than fixed on the class."""

    method = "anthropic_native_append"

    def __init__(self, model: str) -> None:
        self.system_base = prompt.STRICT_NOCOT_PROMPT if model in _NATIVE_ADAPTIVE_THINKING else None
        self.no_cot_enforcement = "append 'Answer:' + thinking omitted (or adaptive-minimum + strict prompt)"

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        tail = messages[-1]
        msgs = list(messages[:-1]) + [{"role": "user", "content": f"{tail['content']}\n{PREFILL}"}]
        params = {"model": _native_model_id(model), "system": system, "messages": msgs,
                  "max_tokens": max_tokens}
        params.update(_native_thinking_params(model))
        return params


class AnthropicNativeToolBackend(AnthropicNativeBackend):
    """Native forced tool call: `tool_choice` forces the `submit_answer` tool (Anthropic's own
    tool_choice/tool shape — `_TOOLS` is already in this shape, unlike the OpenAI-style wrapping
    `OpenRouterToolBackend` needs). No strict system prompt (unlike `AnthropicNativeAppendBackend`
    on adaptive-only ids) — same reasoning as `OpenRouterToolBackend`: the forced tool call itself
    is what structurally prevents free-text output, so both model kinds share the dataset's plain
    prompt here; only the append channel (no structural constraint) needs the strict fallback."""

    method = "anthropic_native_tool"
    no_cot_enforcement = ("forced tool use (submit_answer, native tool_choice) + thinking omitted "
                          "(or adaptive-minimum on adaptive-only ids) — answer as a tool call, no free text")

    def __init__(self, answer_schema: str = "integer", model: str = ""):
        self._tool = _TOOLS[answer_schema]
        # The explicit-disable variant is a DIFFERENT request shape than what this label
        # historically collected (opus-5's stored anthropic_native_tool rows were gathered
        # under the pre-2026-08-04 adaptive+low-effort params), so it keys under its own
        # label — the store must never hold two request shapes behind one signature.
        if model in _NATIVE_EXPLICIT_DISABLE:
            self.method = "anthropic_native_tool_nothink"

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        params = {"model": _native_model_id(model), "system": system, "messages": list(messages),
                  "max_tokens": max(max_tokens, STRUCTURED_MIN_TOKENS),
                  "tools": [{"name": self._tool["name"], "description": self._tool["description"],
                             "input_schema": self._tool["input_schema"]}],
                  "tool_choice": {"type": "tool", "name": self._tool["name"]}}
        params.update(_native_thinking_params(model))
        return params

    def extract_tool_input(self, resp: Any) -> Any:
        for block in resp.content:
            if block.type == "tool_use" and block.name == self._tool["name"]:
                return block.input if isinstance(block.input, dict) else None
        return None

    def extract_text(self, resp: Any) -> str:
        obj = self.extract_tool_input(resp)
        return "" if not obj else str(obj.get("answer", ""))

    def tool_violation(self, resp: Any):
        return _dict_tool_violation(self.extract_tool_input(resp), resp.stop_reason == "max_tokens")


_VALID_METHODS = ("prefill", "prefill_closed", "append", "structured")


_VALID_TRANSPORTS = ("openrouter", "anthropic_native")


def backend_for(model: str, method: str, answer_schema: str = "integer", *,
                transport: str = "openrouter"):
    """Resolve (model id, no-CoT method, transport) to a backend instance (the single routing
    point).

    Every model here is an OpenRouter namespaced id (a "/" in it) — `transport` picks whether it's
    actually reached via OpenRouter (default) or api.anthropic.com directly (`anthropic_native`,
    "anthropic/..." ids only). The method is explicit per call — how it is CHOSEN per (model,
    dataset) is `registry.resolve_method`'s job; an impossible combination (e.g. prefill on a
    model that doesn't support it) raises rather than silently falling back, so a cell's method
    label always names exactly what ran. `answer_schema` (from the dataset) selects the
    structured backends' integer vs string shape.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}.")
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(f"transport must be one of {_VALID_TRANSPORTS}, got {transport!r}.")
    if "/" not in model:
        raise ValueError(f"{model!r}: expected an OpenRouter namespaced id (e.g. 'openai/gpt-5.5').")

    if transport == "anthropic_native":
        if not model.startswith("anthropic/"):
            raise ValueError(f"{model!r}: anthropic_native transport is for anthropic/* ids only.")
        if method == "structured":
            return AnthropicNativeToolBackend(answer_schema, model)
        if method == "append":
            return AnthropicNativeAppendBackend(model)
        if method == "prefill_closed":
            raise ValueError(f"{model!r}: 'prefill_closed' is an OpenRouter-transport channel "
                             "(probed per OR endpoint) — not available on anthropic_native.")
        if model in _SUPPORTS_PREFILL:
            return AnthropicNativePrefillBackend()
        raise ValueError(f"{model!r}: this model cannot 'prefill' — use 'append' or 'structured'.")

    if method == "prefill_closed":
        if model not in _SUPPORTS_CLOSED_PREFILL:
            raise ValueError(f"{model!r}: closed-think prefill is probe-verified only for "
                             f"{sorted(_SUPPORTS_CLOSED_PREFILL)} — accepting a trailing "
                             "assistant message does not imply the closed think block works.")
        return OpenRouterClosedThinkPrefillBackend(model)
    if method == "structured":
        # Claude aliases use a forced tool call (mirrors the mechanics this repo studies for the
        # adaptive-only model); other namespaces use a strict JSON schema.
        if model.startswith("anthropic/"):
            return OpenRouterToolBackend(answer_schema)
        return OpenRouterStructuredBackend(answer_schema)
    if method == "append":
        if model in _MANDATORY_REASONING:
            return OpenRouterAdaptiveBackend()
        return OpenRouterBackend()
    # Prefill: only Claude ids that honor a trailing assistant message as a continuation.
    if model in _SUPPORTS_PREFILL:
        return OpenRouterPrefillBackend(model)
    raise ValueError(f"{model!r}: this model cannot 'prefill' — use 'append' or 'structured'.")


def request_params(model: str, method: str, pool: List[Dict[str, Any]], item: Dict[str, Any],
                   cond: conditions.Condition, ds: Any, match_demos: bool = False, *,
                   transport: str = "openrouter") -> Dict[str, Any]:
    """Build one cell's request params: render the problem, assemble the prompt, apply the
    backend's no-CoT enforcement. `ds` is a `registry.Dataset` (untyped here so backends never
    imports registry) supplying the instruction, filler wording, token cap, and answer schema.
    `match_demos`: condition-matched few-shot opt-in — see `prompt.build_messages`'s `demo_cond`
    doc; False (default) is the paper-faithful behavior. `transport`: see `backend_for`.
    Shared by the sweep's cell builder and the cost estimator; the live smoke check reaches the
    same `backend.build_params` via `backend.complete()`."""
    backend = backend_for(model, method, ds.answer_schema, transport=transport)
    text = conditions.render(item["problem"], cond)
    messages = prompt.build_messages(pool, text, backend, demo_cond=cond if match_demos else None)
    # A backend with its own system_base (the adaptive strict no-CoT prompt) still defers to the
    # dataset's strict variant when one is set — same anti-think scaffold, answer format adapted
    # to the dataset's answer type.
    if backend.system_base is not None:
        base = getattr(ds, "strict_prompt", None) or backend.system_base
    else:
        base = ds.system_prompt
    system = prompt.system_for(cond, base=base, suffix=ds.filler_suffix)
    return backend.build_params(model, messages, system=system, max_tokens=ds.max_answer_tokens)
