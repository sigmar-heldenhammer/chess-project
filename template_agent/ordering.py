# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:26:20 2026

@author: Judson
"""

import chess
from typing import Optional, Sequence, List, Dict, Tuple, defaultdict
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



class HistoryOrdering(ActivityOrdering):
    """
    Activity-based ordering with a history-heuristic tie-break.

    Mirrors the standalone HistoryAgent behavior:
      - Maintain separate history tables for White and Black (per game).
      - On a maximizing-node beta cutoff (Option B), increment history for the cutoff-causing move
        for the side to move at that node.
      - When ordering moves, sort by:
          (1) activity_score desc
          (2) history_count desc (for side to move at this node)
    """

    def __init__(self):
        # Per-game history tables, keyed by move.uci()
        self._hist_white: Dict[str, int] = defaultdict(int)
        self._hist_black: Dict[str, int] = defaultdict(int)

        # Track whether we have seen at least one "non-empty" game so we can clear
        # tables when a brand new game starts (move_stack empties).
        self._history_game_active = False

    # --- per-game lifecycle helpers (best-effort) ---

    def _maybe_reset_for_new_game(self, board: chess.Board) -> None:
        """
        Clears history when we detect a new game.

        We treat an empty move_stack as a new game boundary. This is safe because during search
        we use push/pop; move_stack won't become empty except at a true new game root.
        """
        if len(board.move_stack) == 0:
            if self._history_game_active:
                self._hist_white.clear()
                self._hist_black.clear()
            self._history_game_active = True

    def _table_for_turn(self, turn: chess.Color) -> Dict[str, int]:
        return self._hist_white if turn == chess.WHITE else self._hist_black

    def _history_count(self, turn: chess.Color, mv: chess.Move) -> int:
        return int(self._table_for_turn(turn).get(mv.uci(), 0))

    def _bump_history(self, turn: chess.Color, mv: chess.Move) -> None:
        self._table_for_turn(turn)[mv.uci()] += 1

    # --- required OrderingPolicy methods ---

    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        # Reset tables between games (best effort).
        self._maybe_reset_for_new_game(board)

        turn = board.turn

        # Score each candidate move once.
        scored: List[Tuple[float, int, chess.Move]] = []
        for mv in moves:
            a = self._activity_score(board, mv)
            h = self._history_count(turn, mv)
            scored.append((a, h, mv))

        # Primary: activity desc; Secondary: history desc
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        ordered = [mv for _, _, mv in scored]

        # Optional TT hint: force first (helps if TT stores best_move later).
        if tt_move_hint is not None and tt_move_hint in ordered:
            ordered.remove(tt_move_hint)
            ordered.insert(0, tt_move_hint)

        return ordered

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: "CutoffKind",
        depth: int,
    ) -> None:
        """
        Option B: update history ONLY on maximizing-node beta cutoffs.

        In the framework search loop, we call:
          - cutoff="beta" at maximizing nodes (beta cut)
          - cutoff="alpha" at minimizing nodes (alpha cut)

        So we bump history only when cutoff == "beta".
        """
        if cutoff != "beta":
            return

        # board.turn is the side-to-move at this node (board is popped back to node state).
        self._bump_history(board.turn, move)
