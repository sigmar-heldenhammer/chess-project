# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 21:11:50 2026

@author: Judson
"""

from typing import Optional, Protocol, Sequence, List, Literal
import chess
from dataclasses import dataclass


# ----------------------------
# Shared helpers / constants
# ----------------------------

PIECE_VALUES: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
MATE_VALUE = 1000.0


def material_balance(board: chess.Board, root_color: chess.Color) -> float:
    """Material from root_color perspective, in pawns."""
    score = 0.0
    for piece_type, val in PIECE_VALUES.items():
        score += val * (
            len(board.pieces(piece_type, root_color))
            - len(board.pieces(piece_type, not root_color))
        )
    return score


# ----------------------------
# Search context and results
# ----------------------------

CutoffKind = Literal["none", "alpha", "beta"]
TTFlag = Literal["EXACT", "LOWER", "UPPER"]


@dataclass(frozen=True)
class SearchContext:
    board: chess.Board
    root_color: chess.Color
    maximizing: bool
    depth: int
    alpha: float
    beta: float
    # optional timing hooks (iterative deepening / depth adjustment later)
    time_left: Optional[float] = None
    orig_time: Optional[float] = None
    ply_from_root: int = 0


@dataclass(frozen=True)
class SearchResult:
    value: float
    pv: List[chess.Move]
    cutoff: CutoffKind
    best_move: Optional[chess.Move]


@dataclass(frozen=True)
class TTProbe:
    """Result of probing TT."""
    hit: bool
    value: Optional[float] = None
    flag: Optional[TTFlag] = None
    best_move_hint: Optional[chess.Move] = None
    stored_depth: int = 0


# ----------------------------
# Strategy interfaces (Protocols)
# ----------------------------
class Evaluator(Protocol):
    def evaluate(self, board: chess.Board, root_color: chess.Color) -> float: ...


class TerminalPolicy(Protocol):
    def terminal_value(self, board: chess.Board, root_color: chess.Color, evaluator: Evaluator) -> Optional[float]:
        """
        Return a numeric score if terminal (mate/draw/claim), else None.
        Keep terminal logic *history-aware* here (claim_draw=True).
        """


class LeafPolicy(Protocol):
    def leaf_value(
        self,
        ctx: SearchContext,
        evaluator: Evaluator,
    ) -> Optional[float]:
        """
        Return a numeric score if the node should stop expanding due to depth==0
        (or quiescence, etc.), else None.
        """


class OrderingPolicy(Protocol):
    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
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
        """Called when an alpha/beta cutoff occurs (history heuristic hook)."""


class TranspositionTable(Protocol):
    def probe(self, board: chess.Board, *, depth: int, alpha: float, beta: float) -> TTProbe: ...
    def store(
        self,
        board: chess.Board,
        *,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None: ...


class DepthPolicy(Protocol):
    def effective_depth(
        self,
        *,
        depth: int,
        orig_time: Optional[float],
        time_left: Optional[float],
    ) -> int:
        ...
