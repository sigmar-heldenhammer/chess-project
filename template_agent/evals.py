# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:42:56 2026

@author: Judson
"""
import chess
from .agent_templates import material_balance


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
