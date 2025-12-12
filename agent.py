# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:47:00 2025

@author: Judson
"""

# agents.py
import chess
from typing import Optional

class Agent:
    def select_move(
        self,
        board: chess.Board,
        *,
        time_left: Optional[float] = None,   # seconds left on clock (optional)
        orig_time: Optional[float] = None,   # total seconds available for the game (optional)
        increment: float = 0.0,              # increment per move (optional)
        move_number: int = 1,                # 1-based fullmove number
        color: chess.Color = chess.WHITE,    # True = White, False = Black
    ) -> chess.Move:
        """Return a legal move for the given board."""
        raise NotImplementedError
