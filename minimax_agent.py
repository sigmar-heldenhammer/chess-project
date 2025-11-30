# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 21:05:46 2025

@author: Judson
"""

import math
import random
import chess
from typing import Optional, Tuple
from agents import Agent

# Simple material values (kings excluded)
PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0,
}

MATE_VALUE = 1_000_000.0  # large magnitude so mate dominates material

def _material_balance(board: chess.Board, color: chess.Color) -> float:
    """(my material) - (their material)."""
    opp = not color
    score = 0.0
    for p, v in PIECE_VALUES.items():
        if p == chess.KING:
            continue
        score += v * (len(board.pieces(p, color)) - len(board.pieces(p, opp)))
    return score

def _evaluate(board: chess.Board, root_color: chess.Color) -> float:
    """Static eval from the perspective of root_color. Positive is good for root_color."""
    if board.is_game_over(claim_draw=True):
        res = board.result(claim_draw=True)
        if res == "1-0":
            return MATE_VALUE if root_color == chess.WHITE else -MATE_VALUE
        if res == "0-1":
            return MATE_VALUE if root_color == chess.BLACK else -MATE_VALUE
        return 0.0  # draw
    return _material_balance(board, root_color)

class MinimaxAgent(Agent):
    """
    Depth-limited minimax with optional alpha-beta pruning and simple move ordering.
    """

    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
    ):
        assert depth >= 1
        self.depth = depth
        self.rand = random.Random(seed)
        self.use_alpha_beta = use_alpha_beta
        self.order_moves = order_moves


    def _evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        return _evaluate(board, root_color)
    
    # ---- public API expected by the arena ----
    def select_move(self, board: chess.Board, **ctx) -> chess.Move:
        root_color: chess.Color = ctx.get("color", board.turn)

        best_val = -math.inf
        best_moves = []

        moves = list(board.legal_moves)
        if self.order_moves:
            moves = self._ordered_moves(board, moves)

        alpha, beta = -math.inf, math.inf

        for mv in moves:
            board.push(mv)
            val = self._search(
                board=board,
                depth=self.depth - 1,
                root_color=root_color,
                maximizing=False,  # opponent's turn
                alpha=alpha,
                beta=beta,
            )
            board.pop()

            if val > best_val + 1e-9:
                best_val = val
                best_moves = [mv]
            elif abs(val - best_val) <= 1e-9:
                best_moves.append(mv)

            if self.use_alpha_beta:
                # Alpha from the root's perspective (we’re at a max layer)
                alpha = max(alpha, best_val)

        # Tie-break randomly to avoid deterministic play when equal
        return self.rand.choice(best_moves) if best_moves else next(iter(board.legal_moves))

    # ---- core search ----
    def _search(
        self,
        board: chess.Board,
        depth: int,
        root_color: chess.Color,
        maximizing: bool,
        alpha: float,
        beta: float,
    ) -> float:
        # Terminal node or depth cutoff
        if depth == 0 or board.is_game_over(claim_draw=True):
            # small mate-distance tweak: prefer faster mates / avoid slower losses
            base = self._evaluate(board, root_color)
            if abs(base) >= MATE_VALUE:
                # The deeper we are, the smaller the adjustment (closer mate is better)
                sign = 1.0 if base > 0 else -1.0
                return sign * (MATE_VALUE - (self.depth - depth))
            return base

        legal = list(board.legal_moves)
        if not legal:
            # no legal moves—should have been caught by is_game_over, but safe:
            return self._evaluate(board, root_color)

        if self.order_moves:
            legal = self._ordered_moves(board, legal)

        if maximizing:
            value = -math.inf
            for mv in legal:
                board.push(mv)
                child = self._search(board, depth - 1, root_color, False, alpha, beta)
                board.pop()
                value = max(value, child)
                if self.use_alpha_beta:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break  # beta cut
            return value
        else:
            value = math.inf
            for mv in legal:
                board.push(mv)
                child = self._search(board, depth - 1, root_color, True, alpha, beta)
                board.pop()
                value = min(value, child)
                if self.use_alpha_beta:
                    beta = min(beta, value)
                    if alpha >= beta:
                        break  # alpha cut
            return value

    # ---- simple move ordering heuristics ----
    def _ordered_moves(self, board: chess.Board, moves) -> list[chess.Move]:
        """
        Very light ordering: captures first (MVV-LVA-ish), then promotions,
        then checks, then the rest. This helps alpha-beta a lot.
        """
        scored: list[Tuple[float, chess.Move]] = []
        for mv in moves:
            score = 0.0
            if board.is_capture(mv):
                victim = board.piece_at(mv.to_square)
                attacker = board.piece_at(mv.from_square)
                v = PIECE_VALUES.get(victim.piece_type, 0.0) if victim else 0.0
                a = PIECE_VALUES.get(attacker.piece_type, 0.0) if attacker else 0.0
                score += 10.0 * v - 0.1 * a  # MVV-LVA-ish
            if mv.promotion:
                score += 5.0 + PIECE_VALUES.get(mv.promotion, 0.0)
            # Quick check bonus (requires push/pop to test safely)
            board.push(mv)
            if board.is_check():
                score += 0.5
            board.pop()
            scored.append((score, mv))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [mv for _, mv in scored]
