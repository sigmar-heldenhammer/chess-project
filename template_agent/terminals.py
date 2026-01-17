# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:44:04 2026

@author: Judson
"""

import chess
from .agent_templates import MATE_VALUE, material_balance, Evaluator
from typing import Optional

class DefaultTerminal:
    """
    Terminal check using python-chess' claim_draw=True.
    Uses evaluator material to apply optional draw contempt logic.
    """
    def __init__(self, *, mate_value: float = MATE_VALUE, draw_contempt: float = 0.0):
        self.mate_value = float(mate_value)
        self.draw_contempt = float(draw_contempt)

    def terminal_value(self, board: chess.Board, root_color: chess.Color, evaluator: Evaluator) -> Optional[float]:
        if not board.is_game_over(claim_draw=True):
            return None

        res = board.result(claim_draw=True)
        if res == "1-0":
            return self.mate_value if root_color == chess.WHITE else -self.mate_value
        if res == "0-1":
            return self.mate_value if root_color == chess.BLACK else -self.mate_value

        # Draw: optionally bias using contempt.
        if self.draw_contempt == 0.0:
            return 0.0

        mat = material_balance(board, root_color)
        # If we're better, dislike draws (negative). If worse, like draws (positive).
        return (-abs(self.draw_contempt)) if mat >= 0.0 else (+abs(self.draw_contempt))
