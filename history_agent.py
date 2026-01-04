# -*- coding: utf-8 -*-
"""
History-heuristic + Transposition-table agent.

Implements a simple history heuristic:
- Maintain separate history tables for White and Black (per game).
- When a maximizing-node beta cutoff occurs (Option B), increment the history
  count for the cutoff-causing move for the side to move at that node.
- When ordering moves, sort primarily by activity score, and break ties using
  the history count (higher first).

Notes:
- Uses move.uci() as the history key (fast + unambiguous).
- History is cleared at the start of each new game (when board.move_stack is empty).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, List, Tuple

import chess
import os
from datetime import datetime

from tt_agent import TTAgent


class HistoryAgent(TTAgent):
    """
    TTAgent + history heuristic for tie-breaking in move ordering.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Per-game history tables; keys are move.uci() strings, values are counts.
        self._history_white = defaultdict(int)
        self._history_black = defaultdict(int)

        # Track whether we have "attached" history to the current game yet.
        # We use board.move_stack == [] as the boundary between games.
        self._history_game_active = False

    # --- history helpers ---


    def clear_history(self):
        """
        Write the current history tables to disk for inspection,
        then clear them for the next game.
        """

        self.dump_table(self._history_white, "white")
        self.dump_table(self._history_black, "black")
    
        self.history_white.clear()
        self.history_black.clear()
        
    def dump_table(self, table, side_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("history_dumps", exist_ok=True)
        
        if not table:
            return
        path = os.path.join(
            "history_dumps",
            f"history_{side_name}_{timestamp}.txt"
        )
        with open(path, "w", encoding="utf-8") as f:
            for move, count in sorted(
                table.items(), key=lambda x: x[1], reverse=True
            ):
                f.write(f"{move}: {count}\n")
    


    def _maybe_reset_history_for_new_game(self, board: chess.Board) -> None:
        """
        Reset history at the beginning of a new game.

        We treat an empty move_stack as a reliable signal that this is a fresh game.
        """
        if len(board.move_stack) == 0:
            print("test")
            # If we're at the start position (or start of a new game),
            # clear any accumulated history from prior games.
            if self._history_game_active:
                self.clear_history()
            # Mark active so mid-search push/pop doesn't matter (this is only called at root).
            self._history_game_active = True

    def _hist_table_for_turn(self, turn: chess.Color):
        return self._history_white if turn == chess.WHITE else self._history_black

    def _bump_history(self, turn: chess.Color, mv: chess.Move) -> None:
        # Key by UCI (fast, stable, includes promotion piece)
        self._hist_table_for_turn(turn)[mv.uci()] += 1

    def _history_count(self, turn: chess.Color, mv: chess.Move) -> int:
        return int(self._hist_table_for_turn(turn).get(mv.uci(), 0))

    # --- public API override ---

    def select_move(
        self,
        board: chess.Board,
        *,
        color: chess.Color | None = None,
        time_left: Optional[float] = None,
        orig_time: Optional[float] = None,
        **kwargs
    ) -> chess.Move:
        # Clear per-game history when we detect a new game at the root call.
        self._maybe_reset_history_for_new_game(board)

        
        #self.dump_table(self._history_white, "white")
        #self.dump_table(self._history_black, "black")
        return super().select_move(board, color=color, time_left=time_left, orig_time=orig_time, **kwargs)

    # --- move ordering override ---

    def _ordered_moves(self, board: chess.Board, moves) -> List[chess.Move]:
        """
        Order moves by:
          1) descending activity score (same as MinimaxAgent)
          2) descending history count for the side to move at this node

        Moves tied on both keys are left in arbitrary order.
        """
        activity = self._move_activity(board)
        score_map = {mv: sc for mv, sc in activity}

        turn = board.turn
        scored: List[Tuple[float, int, chess.Move]] = [
            (score_map.get(mv, 0.0), self._history_count(turn, mv), mv) for mv in moves
        ]
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [mv for _, _, mv in scored]

    # --- core search override (adds history updates) ---

    def _search(
        self,
        board: chess.Board,
        depth: int,
        root_color: chess.Color,
        maximizing: bool,
        alpha: float,
        beta: float,
    ):
        """
        Same as TTAgent._search, but with history updates on maximizing-node cutoffs (Option B).
        """
# =============================================================================
#         if depth == self.depth:
#             white_len = len(self._history_white)
#             black_len = len(self._history_black)
#             print(f"white: {white_len}")
#             print(f"black: {black_len}")
# =============================================================================
        
        
        # 1) TT probe (only for positive depths; leaf evals are cheap and depth-dependent)
        if self.tt is not None and depth > 0:
            self.tt_lookups += 1
            key = self._tt_key(board)
            entry = self.tt.get(key)
            if entry is not None and entry.depth >= depth:
                self.tt_hits += 1
                return entry.score, []

        # 2) Leaf / terminal
        if depth <= 0 or board.is_game_over():
            return self._evaluate(board, root_color), []

        legal = list(board.legal_moves)
        if not legal:
            return self._evaluate(board, root_color), []

        if self.order_moves:
            legal = self._ordered_moves(board, legal)

        # Track whether we experienced a cutoff. If so, we won't store this node as "exact".
        cutoff = False

        if maximizing:
            best_val = -math.inf
            best_pv: List[chess.Move] = []

            for mv in legal:
                board.push(mv)
                child_val, child_pv = self._search(board, depth - 1, root_color, False, alpha, beta)
                board.pop()

                if child_val > best_val:
                    best_val = child_val
                    best_pv = [mv] + child_pv

                if self.use_alpha_beta:
                    if best_val > alpha:
                        alpha = best_val
                    if alpha >= beta:
                        cutoff = True
                        # Option B: update history on maximizing-node cutoffs only.
                        # At a maximizing node, board.turn is the side to move for this node.
                        self._bump_history(board.turn, mv)
                        break  # beta cutoff

            # Store if fully searched (no cutoff) and TT enabled
            if self.tt is not None and not cutoff:
                key = self._tt_key(board)
                from tt_agent import TTEntry
                self.tt[key] = TTEntry(depth=depth, score=best_val)
                self.tt_stores += 1

            return best_val, best_pv

        else:
            best_val = math.inf
            best_pv: List[chess.Move] = []

            for mv in legal:
                board.push(mv)
                child_val, child_pv = self._search(board, depth - 1, root_color, True, alpha, beta)
                board.pop()

                if child_val < best_val:
                    best_val = child_val
                    best_pv = [mv] + child_pv

                if self.use_alpha_beta:
                    if best_val < beta:
                        beta = best_val
                    if alpha >= beta:
                        cutoff = True
                        break  # alpha cutoff

            if self.tt is not None and not cutoff:
                key = self._tt_key(board)
                from tt_agent import TTEntry
                self.tt[key] = TTEntry(depth=depth, score=best_val)
                self.tt_stores += 1

            return best_val, best_pv
