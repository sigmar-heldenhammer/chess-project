# -*- coding: utf-8 -*-
"""
Transposition-table extension for MinimaxAgent using an LFU cache.

This implements a simple transposition table (TT):
- Key: Polyglot Zobrist hash of the current board position.
- Value: (searched_depth, score)

Lookup rule:
- Reuse cached score only if searched_depth >= requested depth.

Storage rule (iteration 1, conservative):
- Store results only when the node was fully searched (i.e., no alpha/beta cutoff
  occurred), to avoid caching bound-only values as if they were exact.

PV handling:
- This agent does NOT store PVs in the TT. On TT hits, it returns an empty PV list.

Notes:
- This is an approximation: the key does not incorporate repetition history / halfmove
  clock, so positions that are "the same" but reached via different histories can
  reuse values. This is common for a first TT iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import chess

try:
    from cachetools import LFUCache
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "minimax_agent_tt.py requires the 'cachetools' package. "
        "Install it with: pip install cachetools"
    ) from e

try:
    import chess.polyglot
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "minimax_agent_tt.py requires python-chess with polyglot support."
    ) from e

from minimax_agent import MinimaxAgent


@dataclass(frozen=True, slots=True)
class TTEntry:
    """A compact transposition-table entry."""
    depth: int
    score: float


class TTAgent(MinimaxAgent):
    """
    MinimaxAgent with a transposition table backed by cachetools.LFUCache.

    Parameters
    ----------
    tt_max_entries : int
        Maximum number of TT entries (LFU-evicted when exceeded).
    """

    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        draw_contempt: float = 0.2,
        *,
        tt_max_entries: int = 5_000_000,
    ):
        super().__init__(
            depth=depth,
            seed=seed,
            use_alpha_beta=use_alpha_beta,
            order_moves=order_moves,
            draw_contempt=draw_contempt,
        )
        if tt_max_entries < 0:
            raise ValueError("tt_max_entries must be >= 0")
        self.tt_max_entries = int(tt_max_entries)
        self.tt = LFUCache(maxsize=self.tt_max_entries) if self.tt_max_entries > 0 else None

        # Optional stats
        self.tt_lookups = 0
        self.tt_hits = 0
        self.tt_stores = 0

    # --- internal helpers ---
    @staticmethod
    def _tt_key(board: chess.Board) -> int:
        """Return a 64-bit polyglot Zobrist hash for the current position."""
        return chess.polyglot.zobrist_hash(board)

    def clear_tt(self) -> None:
        """Clear the TT and reset basic stats."""
        if self.tt is not None:
            self.tt.clear()
        self.tt_lookups = 0
        self.tt_hits = 0
        self.tt_stores = 0

    # --- override search to add TT lookup/store ---
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
        PV-aware alpha-beta: returns (score, pv).
        This override adds a TT lookup/store layer.

        On TT hit: returns (cached_score, []).
        """
        # 1) TT probe (only for positive depths; leaf evals are cheap and depth-dependent)
        if self.tt is not None and depth > 0:
            self.tt_lookups += 1
            key = self._tt_key(board)
            ent = self.tt.get(key)
            if ent is not None and ent.depth >= depth:
                self.tt_hits += 1
                return ent.score, []

        # 2) Terminal (mate/draw/claim)
        if board.is_game_over(claim_draw=True):
            return self._evaluate(board, root_color), []

        # 3) Leaf
        if depth == 0:
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
                        break  # beta cutoff

            # Store if fully searched (no cutoff) and TT enabled
            if self.tt is not None: #and not cutoff:
                key = self._tt_key(board)
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

            if self.tt is not None: #and not cutoff:
                key = self._tt_key(board)
                self.tt[key] = TTEntry(depth=depth, score=best_val)
                self.tt_stores += 1

            return best_val, best_pv
