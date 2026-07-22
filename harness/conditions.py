"""Render the problem portion of the prompt under each condition.

Three conditions, all operating on the final user turn (the demonstrations are unchanged, unless
condition-matched — see `prompt.build_messages`'s `demo_cond` parameter):

  baseline        : the problem on its own.
  repeat(r)       : r labeled copies of the problem, one per line:
                      Problem: {p}
                      Problem (repeat 2): {p}
                      ... up to (repeat r).
                    r == 1 renders just "Problem: {p}" (so the repeat anchor is labeled, which
                    differs slightly from the bare-problem baseline — matching the paper).
  filler(n)       : "Problem: {p}" then a line "Filler: 1 2 3 ... n" (n == 0 is the baseline).
                    The instruction itself also gains a sentence announcing the filler — that
                    lives in the system prompt (see prompt.system_for), not here.

Repeated copies and filler are placed where the model reads them before answering, giving it
more forward-pass compute without any chain-of-thought.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    kind: str          # "baseline" | "repeat" | "filler"
    value: int = 0     # r for repeat, n for filler; ignored for baseline

    @property
    def label(self) -> str:
        if self.kind == "baseline":
            return "baseline"
        if self.kind == "repeat":
            return f"repeat_r{self.value}"
        if self.kind == "filler":
            return f"filler_f{self.value}"
        raise ValueError(f"unknown condition kind: {self.kind}")


def baseline() -> Condition:
    return Condition("baseline")


def repeat(r: int) -> Condition:
    return Condition("repeat", r)


def filler(n: int) -> Condition:
    return Condition("filler", n)


def render(problem: str, cond: Condition) -> str:
    """Return the user-turn text for `problem` under `cond`."""
    if cond.kind == "baseline":
        return problem
    if cond.kind == "repeat":
        copies = max(1, cond.value)
        parts = [f"Problem: {problem}"]
        for k in range(2, copies + 1):
            parts.append(f"Problem (repeat {k}): {problem}")
        return "\n".join(parts)
    if cond.kind == "filler":
        if cond.value <= 0:
            return problem
        numbers = " ".join(str(i) for i in range(1, cond.value + 1))
        return f"Problem: {problem}\nFiller: {numbers}"
    raise ValueError(f"unknown condition kind: {cond.kind}")
