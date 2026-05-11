

# -*- coding: utf-8 -*-
"""
criteria.py

Reusable board-evaluation criteria functions + an explicit registry.

Each criterion returns a score where "higher is better for root_color".
Most criteria are designed to produce values in (approximately) [-1, 1].
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple
import chess

from .agent_templates import PIECE_VALUES, material_balance


CriterionFn = Callable[[chess.Board, chess.Color], float]


def _central_squares() -> Tuple[int, ...]:
    """Squares for the 4x4 center c3..f6 (files c-f, ranks 3-6)."""
    files = ["c", "d", "e", "f"]
    ranks = ["3", "4", "5", "6"]
    return tuple(chess.parse_square(f + r) for f in files for r in ranks)


_CENTER_SQUARES: Tuple[int, ...] = _central_squares()


def crit_material_share(board: chess.Board, root_color: chess.Color) -> float:
    """
    Material share mapped to [-1, 1]:
      share = friendly_value / total_value in [0, 1]
      return 2*share - 1

    Uses PIECE_VALUES from agent_templates. Kings are not included there.
    """
    def side_value(color: chess.Color) -> float:
        s = 0.0
        for ptype, val in PIECE_VALUES.items():
            s += val * len(board.pieces(ptype, color))
        return s

    friendly = side_value(root_color)
    hostile = side_value(not root_color)
    total = friendly + hostile
    if total <= 0.0:
        return 0.0  # only kings; neutral
    share = friendly / total
    return 2.0 * share - 1.0


def crit_material_balance(board: chess.Board, root_color: chess.Color) -> float:
    """
    just use the standard material balance
    """
    score = material_balance(board, root_color)


    return score

def crit_center_control(board: chess.Board, root_color: chess.Color) -> float:
    """
    (#friendly pieces in c3..f6 - #hostile pieces in c3..f6) / 16 in [-1, 1].
    """
    friendly = 0
    hostile = 0
    for sq in _CENTER_SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        if piece.color == root_color:
            friendly += 1
        else:
            hostile += 1
    return (friendly - hostile) / 16.0


def crit_pseudo_active_pieces(board: chess.Board, root_color: chess.Color) -> float:
    """
    Copied from evaluation_agent_decorator's pseudo-active-pieces notion:

    Counts root_color pieces where the number of attacked squares that are NOT
    occupied by own pieces is >= 3. Normalized by /8.

    NOTE: This criterion also incorporates side-to-move by negating when
    board.turn != root_color (as in the reference implementation).
    """
    occ_own = board.occupied_co[root_color]
    active = 0.0

    for sq, p in board.piece_map().items():
        if p.color != root_color:
            continue
        if len(board.attacks(sq) & ~occ_own) >= 3:
            active += 1.0

    active = active / 8.0
    return float(active) if board.turn == root_color else float(-1.0 * active)



def crit_pawn_structure(board: chess.Board, root_color: chess.Color) -> float:
    """
    Fraction of friendly pawns defended by friendly pawns minus fraction of
    hostile pawns defended by hostile pawns.

    Fast bitboard implementation. Returns approximately [-1, 1].
    """
    def defended_pawn_fraction(color: chess.Color) -> float:
        pawns = board.pawns & board.occupied_co[color]
        pawn_count = pawns.bit_count()
        if pawn_count == 0:
            return 0.0

        if color == chess.WHITE:
            defended = ((pawns << 7) & ~chess.BB_FILE_H) | ((pawns << 9) & ~chess.BB_FILE_A)
        else:
            defended = ((pawns >> 7) & ~chess.BB_FILE_A) | ((pawns >> 9) & ~chess.BB_FILE_H)

        defended_pawns = pawns & defended
        return defended_pawns.bit_count() / pawn_count

    friendly = defended_pawn_fraction(root_color)
    hostile = defended_pawn_fraction(not root_color)
    return friendly - hostile


# ----------------------------
# Explicit registry
# ----------------------------

CRITERIA_REGISTRY: Dict[str, CriterionFn] = {
    "material_share": crit_material_share,
    "material_balance": crit_material_balance,
    "center_control": crit_center_control,
    "pseudo_active_pieces": crit_pseudo_active_pieces,
    "pawn_structure": crit_pawn_structure,
}

# Optional convenience: a canonical ordering for display/debugging
CRITERIA_ORDER: Tuple[str, ...] = (
    "material_share",
    "material_balance",
    "center_control",
    "pseudo_active_pieces",
    "pawn_structure",
)
