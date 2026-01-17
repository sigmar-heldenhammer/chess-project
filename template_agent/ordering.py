# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:26:20 2026

@author: Judson
"""

import chess
from typing import Optional, Sequence, List
from .agent_templates import CutoffKind, PIECE_VALUES

class BasicOrdering:
    """No history; optionally puts TT hint first."""
    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        out = list(moves)
        if tt_move_hint is not None and tt_move_hint in out:
            out.remove(tt_move_hint)
            out.insert(0, tt_move_hint)
        return out

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: CutoffKind,
        depth: int,
    ) -> None:
        return


class ActivityOrdering(BasicOrdering):
    """
    Mirrors your current “activity” heuristic style move ordering:
    captures/promotions/checks first.

    (Still a no-history baseline; history can be layered later by overriding on_cutoff + scoring.)
    """
    def _activity_score(self, board: chess.Board, mv: chess.Move) -> float:
        score = 0.0

        # If in check, prioritize evasions (boost all).
        if board.is_check():
            score += 1.0

        # Captures
        if board.is_capture(mv):
            # victim value
            victim_type = None
            if board.is_en_passant(mv):
                victim_type = chess.PAWN
            else:
                victim = board.piece_at(mv.to_square)
                victim_type = victim.piece_type if victim else None

            attacker = board.piece_at(mv.from_square)
            attacker_type = attacker.piece_type if attacker else None

            victim_val = PIECE_VALUES.get(victim_type, 0.0) if victim_type else 0.0
            attacker_val = PIECE_VALUES.get(attacker_type, 0.0) if attacker_type else 0.0

            score += 10.0 * victim_val - 0.1 * attacker_val

        # Promotions
        if mv.promotion is not None:
            score += PIECE_VALUES.get(mv.promotion, 0.0)

        # Giving check (requires push/pop)
        board.push(mv)
        try:
            if board.is_check():
                score += 0.5
        finally:
            board.pop()

        return score

    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        out = list(moves)
        scored = [(self._activity_score(board, mv), mv) for mv in out]
        scored.sort(key=lambda t: t[0], reverse=True)
        ordered = [mv for _, mv in scored]

        # Optional TT hint: force to front (useful if TT stores best_move).
        if tt_move_hint is not None and tt_move_hint in ordered:
            ordered.remove(tt_move_hint)
            ordered.insert(0, tt_move_hint)
        return ordered
