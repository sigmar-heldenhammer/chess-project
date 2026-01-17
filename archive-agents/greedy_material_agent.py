# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:03:29 2025

@author: Judson
"""

import math, random
import chess
from agents import Agent

# Simple piece values (kings excluded from material eval)
PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0,  # ignored
}

MATE_VALUE = 1_000_000.0  # big number so any mate beats material

class GreedyMaterialAgent(Agent):
    """Naive 1-ply evaluator: terminal > material; chooses move with max eval."""

    def __init__(self, seed: int | None = None):
        self.rand = random.Random(seed)

    def _material_balance(self, board: chess.Board, color: chess.Color) -> float:
        """(my material) - (their material) with simple piece values."""
        score = 0.0
        opp = not color
        for ptype, val in PIECE_VALUES.items():
            if ptype == chess.KING:
                continue
            score += val * (len(board.pieces(ptype, color)) - len(board.pieces(ptype, opp)))
        return score

    def _evaluate(self, board: chess.Board, color: chess.Color) -> float:
        """Positive is good for 'color'."""
        # Terminal outcomes first
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)  # "1-0", "0-1", or "1/2-1/2"
            if res == "1-0":
                return MATE_VALUE if color == chess.WHITE else -MATE_VALUE
            if res == "0-1":
                return MATE_VALUE if color == chess.BLACK else -MATE_VALUE
            return 0.0  # any draw
        # Otherwise, pure material balance
        return self._material_balance(board, color)

    def select_move(self, board: chess.Board, **ctx) -> chess.Move:
        """Pick the legal move that maximizes eval of the resulting position."""
        color: chess.Color = ctx.get("color", board.turn)
        best_val = -math.inf
        best_moves: list[chess.Move] = []

        for move in board.legal_moves:
            board.push(move)
            val = self._evaluate(board, color)
            board.pop()

            # Early exit if immediate mate found
            if val >= MATE_VALUE:
                return move

            if val > best_val + 1e-9:
                best_val = val
                best_moves = [move]
            elif abs(val - best_val) <= 1e-9:
                best_moves.append(move)

        # Tie-break randomly among equally good moves
        return self.rand.choice(best_moves) if best_moves else next(iter(board.legal_moves))
