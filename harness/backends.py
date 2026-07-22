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

# Pin Claude-via-OpenRouter requests to Anthropic's own serving stack. OpenRouter can otherwise
# load-balance "anthropic/..." ids across backing providers per request, which would silently mix
# serving stacks inside one experiment. Applied only to "anthropic/" ids — other namespaces keep
# OpenRouter's default routing.
_ANTHROPIC_PROVIDER_PIN = {"only": ["anthropic"], "allow_fallbacks": False}

# Models verified live to 400 on OpenRouter's unified reasoning-disable ("Reasoning is mandatory
# for this endpoint and cannot be disabled") — these route to OpenRouterAdaptiveBackend under
# `append`, and skip the disable param under `structured` too.
_MANDATORY_REASONING = {"anthropic/claude-fable-5"}

# Claude ids verified live to honor a trailing assistant message as a genuine prefill
# continuation via OpenRouter (adaptive-only ids reject it, mirroring their native behavior).
_SUPPORTS_PREFILL = {"anthropic/claude-opus-4.5"}


def _or_extra_body(model: str, *, reasoning_off: bool = True) -> Dict[str, Any]:
    """The OpenRouter extra_body for `model`: the unified reasoning-disable (skipped for models
    that reject it) plus the Anthropic provider pin for anthropic/* ids. EVERY OpenRouter request
    path builds its extra_body here, so a new backend cannot forget the pin. May return {} —
    callers that must not send an empty extra_body should only attach it when truthy."""
    body: Dict[str, Any] = {"reasoning": {"enabled": False}} if reasoning_off else {}
    if model.startswith("anthropic/"):
        body["provider"] = dict(_ANTHROPIC_PROVIDER_PIN)
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
        import openai
        return openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={"X-Title": "no-cot-harness"},
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
    """Claude via OpenRouter under the prefill method: OpenRouter forwards a trailing assistant
    message to the Anthropic provider as an assistant prefill, so the model continues an
    assistant turn that already says "Answer:" (verified live: continuation arrives without the
    prefix, reasoning_tokens=0). Only meaningful for "anthropic/..." ids — `backend_for` gates the
    route on `_SUPPORTS_PREFILL`."""

    method = "openrouter_prefill"
    no_cot_enforcement = ("prefill 'Answer:' (trailing assistant message via OpenRouter) + "
                          "reasoning {enabled:false} + provider pinned to Anthropic + small max_tokens")

    def build_params(self, model: str, messages: List[Dict[str, Any]], *,
                     system: str, max_tokens: int = MAX_ANSWER_TOKENS) -> Dict[str, Any]:
        msgs = ([{"role": "system", "content": system}] + list(messages)
                + [{"role": "assistant", "content": PREFILL}])
        return {"model": model, "messages": msgs, "max_completion_tokens": max_tokens,
                "extra_body": _or_extra_body(model)}

    def extract_text(self, resp: Any) -> str:
        return PREFILL + (resp.choices[0].message.content or "")


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


_VALID_METHODS = ("prefill", "append", "structured")


def backend_for(model: str, method: str, answer_schema: str = "integer"):
    """Resolve (model id, no-CoT method) to a backend instance (the single routing point).

    Every model here is an OpenRouter namespaced id (a "/" in it). The method is explicit per
    call — how it is CHOSEN per (model, dataset) is `registry.resolve_method`'s job; an
    impossible combination (e.g. prefill on a model that doesn't support it) raises rather than
    silently falling back, so a cell's method label always names exactly what ran.
    `answer_schema` (from the dataset) selects the structured backends' integer vs string shape.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}.")
    if "/" not in model:
        raise ValueError(f"{model!r}: expected an OpenRouter namespaced id (e.g. 'openai/gpt-5.5').")

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
        return OpenRouterPrefillBackend()
    raise ValueError(f"{model!r}: this model cannot 'prefill' — use 'append' or 'structured'.")


def request_params(model: str, method: str, pool: List[Dict[str, Any]], item: Dict[str, Any],
                   cond: conditions.Condition, ds: Any, match_demos: bool = False) -> Dict[str, Any]:
    """Build one cell's request params: render the problem, assemble the prompt, apply the
    backend's no-CoT enforcement. `ds` is a `registry.Dataset` (untyped here so backends never
    imports registry) supplying the instruction, filler wording, token cap, and answer schema.
    `match_demos`: condition-matched few-shot opt-in — see `prompt.build_messages`'s `demo_cond`
    doc; False (default) is the paper-faithful behavior.
    Shared by the sweep's cell builder and the cost estimator; the live smoke check reaches the
    same `backend.build_params` via `backend.complete()`."""
    backend = backend_for(model, method, ds.answer_schema)
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
