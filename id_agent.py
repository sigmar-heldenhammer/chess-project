# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:13:00 2026

@author: Judson
"""

import chess
from history_agent import HistoryAgent
from typing import Optional

class IDAgent(HistoryAgent):
    
    
    
    def select_move(
        self,
        board: chess.Board,
        *,
        color: chess.Color | None = None,
        time_left: Optional[float] = None,
        orig_time: Optional[float] = None,
        **kwargs
    ) -> chess.Move:
        
        best_move = None
        for i in range(1, self.depth):
            best_move = super().select_move(board, color=color, time_left=time_left, orig_time=orig_time, depth=i, **kwargs)
        return best_move
            