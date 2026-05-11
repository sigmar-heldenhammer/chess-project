# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:42:56 2026

@author: Judson
"""
from __future__ import annotations

import chess
from .agent_templates import material_balance
from .criteria import CRITERIA_REGISTRY, CriterionFn

from dataclasses import dataclass
from typing import Dict, List, Tuple

class MaterialEvaluator:
    """Basic evaluator (non-terminal)."""
    def __init__(self, *, draw_contempt: float = 0.0, near_rep_nudge: float = 0.0):
        self.draw_contempt = float(draw_contempt)
        self.near_rep_nudge = float(near_rep_nudge)

    def evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        score = material_balance(board, root_color)

        # Optional "near repetition" soft nudge (kept mild by default).
        if self.near_rep_nudge != 0.0:
            try:
                if board.is_repetition(2) or board.can_claim_threefold_repetition():
                    if score >= 0.0:
                        score -= abs(self.near_rep_nudge)
                    else:
                        score += abs(self.near_rep_nudge)
            except Exception:
                pass

        return score


@dataclass(frozen=True)
class _EnabledCriterion:
    name: str
    weight: float
    fn: CriterionFn


class WeightedCriteriaEvaluator:
    """
    Weighted blend of named criteria from criteria.py.

    - Input: dict[str, float] weights (names must exist in CRITERIA_REGISTRY).
    - Any criterion with weight == 0.0 is skipped entirely (not evaluated).
    - By default we normalize by sum(abs(weights)) so output stays roughly [-1, 1]
      assuming criteria are roughly in that range.
    - Non-terminal only. Terminal scoring should happen in TerminalPolicy.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        *,
        normalize: bool = True,
        clamp: bool = True,
        near_rep_nudge: float = 0.0,
    ):
        if not isinstance(weights, dict):
            raise TypeError("weights must be a dict[str, float]")

        # Validate names early to catch typos.
        unknown = [k for k in weights.keys() if k not in CRITERIA_REGISTRY]
        if unknown:
            known = ", ".join(sorted(CRITERIA_REGISTRY.keys()))
            bad = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown criterion name(s): {bad}. Known: {known}")

        self._normalize = bool(normalize)
        self._clamp = bool(clamp)
        self.near_rep_nudge = float(near_rep_nudge)

        enabled: List[_EnabledCriterion] = []
        for name, w in weights.items():
            w = float(w)
            if w == 0.0:
                continue
            enabled.append(_EnabledCriterion(name=name, weight=w, fn=CRITERIA_REGISTRY[name]))

        # If user passed all zeros, keep an empty list; evaluate() returns 0.0 (plus nudges).
        self._enabled: Tuple[_EnabledCriterion, ...] = tuple(enabled)

        if self._normalize:
            self._denom = sum(abs(c.weight) for c in self._enabled) or 1.0
        else:
            self._denom = 1.0

    def evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        total = 0.0
        for c in self._enabled:
            total += c.weight * float(c.fn(board, root_color))

        score = total / self._denom

        # Optional clamp to contain tiny floating noise, and keep a consistent range.
        if self._clamp:
            if score > 1.0:
                score = 1.0
            elif score < -1.0:
                score = -1.0

        # Optional "near repetition" soft nudge (kept mild by default).
        if self.near_rep_nudge != 0.0:
            try:
                if board.is_repetition(2) or board.can_claim_threefold_repetition():
                    if score >= 0.0:
                        score -= abs(self.near_rep_nudge)
                    else:
                        score += abs(self.near_rep_nudge)
            except Exception:
                pass

        return score
