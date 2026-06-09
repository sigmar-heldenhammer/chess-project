# -*- coding: utf-8 -*-
"""
Move selection policies for the modular minimax framework.

A MoveSelectionPolicy is responsible for deciding which legal moves should be
searched and in what order.

This intentionally separates two ideas:

  - OrderingPolicy: orders a provided sequence of moves.
  - MoveSelectionPolicy: may filter/select a subset of legal moves, then call an
    OrderingPolicy to order the resulting moves.

The search loop can then remain agnostic about specific add-ons such as
quiescence search, while still knowing whether a node was searched selectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Protocol

import chess

from .agent_templates import CutoffKind, OrderingPolicy
from .ordering import ActivityOrdering


@dataclass
class MoveSelection:
    """Result returned by a MoveSelectionPolicy."""
    moves: List[chess.Move]
    is_selective: bool = False


class MoveSelectionPolicy(Protocol):
    def select_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        depth: int,
        *,
        ply_from_root: int = 0,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> MoveSelection:
        ...

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: CutoffKind,
        depth: int,
    ) -> None:
        ...


class DefaultMoveSelection:
    """
    Full-width move selection.

    This preserves the behavior of the previous framework: all legal moves are
    searched, and the configured OrderingPolicy determines only their order.
    """
    def __init__(self, ordering: Optional[OrderingPolicy] = None):
        self.ordering: OrderingPolicy = ordering or ActivityOrdering()

    def select_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        depth: int,
        *,
        ply_from_root: int = 0,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> MoveSelection:
        ordered = self.ordering.order_moves(
            board,
            moves,
            depth,
            tt_move_hint=tt_move_hint,
        )
        return MoveSelection(moves=ordered, is_selective=False)

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: CutoffKind,
        depth: int,
    ) -> None:
        return self.ordering.on_cutoff(
            board=board,
            root_color=root_color,
            move=move,
            cutoff=cutoff,
            depth=depth,
        )


class QuiescentMoveSelection:
    """
    Selective move policy for quiescence-style search.

    Above q_start_depth, this is full-width and delegates to base_ordering.

    At q_start_depth and below, it keeps only tactical moves:
      - captures
      - promotions, if include_promotions=True
      - checking moves, if include_checks=True
      - all legal evasions while in check, if include_evasions=True

    Returning is_selective=True tells the core search loop that stand-pat may be
    considered, subject to generic rules such as "not while in check".
    """
    def __init__(
        self,
        base_ordering: Optional[OrderingPolicy] = None,
        q_start_depth: int = 0,
        include_checks: bool = False,
        include_promotions: bool = True,
        include_evasions: bool = True,
    ):
        self.base_ordering: OrderingPolicy = base_ordering or ActivityOrdering()
        self.q_start_depth = q_start_depth
        self.include_checks = include_checks
        self.include_promotions = include_promotions
        self.include_evasions = include_evasions

    def select_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        depth: int,
        *,
        ply_from_root: int = 0,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> MoveSelection:
        # Full-width phase.
        if depth > self.q_start_depth:
            ordered = self.base_ordering.order_moves(
                board,
                moves,
                depth,
                tt_move_hint=tt_move_hint,
            )
            return MoveSelection(moves=ordered, is_selective=False)

        # If in check and include_evasions is enabled, search every legal evasion.
        # This avoids treating "stand pat" as an option when the side to move is
        # legally required to respond to check.
        if board.is_check() and self.include_evasions:
            ordered = self.base_ordering.order_moves(
                board,
                moves,
                depth,
                tt_move_hint=tt_move_hint,
            )
            return MoveSelection(moves=ordered, is_selective=False)

        tactical: List[chess.Move] = []
        for mv in moves:
            if board.is_capture(mv):
                tactical.append(mv)
            elif self.include_promotions and mv.promotion is not None:
                tactical.append(mv)
            elif self.include_checks:
                board.push(mv)
                try:
                    if board.is_check():
                        tactical.append(mv)
                finally:
                    board.pop()

        ordered = self.base_ordering.order_moves(
            board,
            tactical,
            depth,
            tt_move_hint=None,
        )
        return MoveSelection(moves=ordered, is_selective=True)

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: CutoffKind,
        depth: int,
    ) -> None:
        return self.base_ordering.on_cutoff(
            board=board,
            root_color=root_color,
            move=move,
            cutoff=cutoff,
            depth=depth,
        )
