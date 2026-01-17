# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:47:35 2025

@author: Judson
"""

# random_agent.py
import chess, random
from agents import Agent

class RandomAgent(Agent):
    def __init__(self, seed: int | None = None):
        self.rand = random.Random(seed)

    def select_move(self, board: chess.Board, **kwargs) -> chess.Move:
        return self.rand.choice(list(board.legal_moves))
