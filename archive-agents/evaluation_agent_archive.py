
"""
evaluation_agent.py

A proof-of-concept agent that extends MinimaxAgent and overrides _evaluate
to use a weighted average of simple criteria in the range [-1, 1], while
preserving terminal handling (mate/draw) from the parent.

Criteria implemented:
  1) Material share ratio in [-1, 1]:
     ((friendly_material) / (total_material)) mapped from [0,1] to [-1,1]
     via: 2 * (friendly / total) - 1. (Kings ignored as value 0.)
  2) Central control in [-1, 1]:
     ((#friendly in {c3..f6}) - (#hostile in {c3..f6})) / 16

The final evaluation is the weighted average of these criteria. If no
weights are given, equal weighting is used.
"""

from __future__ import annotations

import chess
from typing import Iterable, List, Optional, Sequence

# We extend the existing MinimaxAgent from minimax_agent.py
from minimax_agent import MinimaxAgent, PIECE_VALUES, MATE_VALUE


def _central_squares() -> list[int]:
    """Squares for the 4x4 center c3..f6 (files c-f, ranks 3-6)."""
    files = ['c', 'd', 'e', 'f']
    ranks = ['3', '4', '5', '6']
    return [chess.parse_square(f + r) for f in files for r in ranks]


class EvaluationAgent(MinimaxAgent):
    """
    Minimax agent with a custom evaluation function that blends multiple
    normalized criteria via weights.
    """

    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(depth=depth, seed=seed, use_alpha_beta=use_alpha_beta, order_moves=order_moves)
        # Default equal weights for two criteria
        self.weights: List[float] = list(weights) if weights is not None else [1.0, 1.0]
        self._center = tuple(_central_squares())

    # --------------- Criteria (each returns value in [-1, 1]) ---------------

    def _crit_material_share(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Material share mapped to [-1, 1]:
            share = friendly_value / total_value in [0,1]
            return 2*share - 1 to get [-1, 1].

        Kings are assigned value 0 in PIECE_VALUES and do not affect totals.
        """
        # Sum material values for both sides
        def side_value(color: chess.Color) -> float:
            s = 0.0
            for ptype, val in PIECE_VALUES.items():
                if val == 0.0:
                    continue
                s += val * len(board.pieces(ptype, color))
            return s

        friendly = side_value(root_color)
        hostile = side_value(not root_color)
        total = friendly + hostile
        if total <= 0.0:
            return 0.0  # only kings on board; neutral
        share = friendly / total  # 0..1
        return 2.0 * share - 1.0   # map to [-1,1]

    def _crit_center_control(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        (#friendly in c3..f6 - #hostile in c3..f6) / 16 in [-1, 1].
        """
        friendly = 0
        hostile = 0
        for sq in self._center:
            piece = board.piece_at(sq)
            if not piece:
                continue
            if piece.color == root_color:
                friendly += 1
            else:
                hostile += 1
        return (friendly - hostile) / 16.0

    # --------------- Weighted combination + terminal handling ---------------

    def _evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Preserve terminal handling (mate/draw) like the parent, then use
        the weighted blend of criteria (each in [-1,1]).
        """
        # Terminal outcomes first (identical to parent semantically)
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
            if res == "1-0":
                return MATE_VALUE if root_color == chess.WHITE else -MATE_VALUE
            if res == "0-1":
                return MATE_VALUE if root_color == chess.BLACK else -MATE_VALUE
            return 0.0  # draw

        # Compute criteria
        c1 = self._crit_material_share(board, root_color)   # [-1,1]
        c2 = self._crit_center_control(board, root_color)   # [-1,1]

        crits = (c1, c2)
        weights = self.weights
        # Pad or trim weights to match number of criteria
        if len(weights) < len(crits):
            weights = list(weights) + [0.0] * (len(crits) - len(weights))
        elif len(weights) > len(crits):
            weights = list(weights[:len(crits)])

        # Weighted average, normalize by sum of abs weights to keep final in [-1,1]
        denom = sum(abs(w) for w in weights) or 1.0
        score = sum(w * v for w, v in zip(weights, crits)) / denom

        # Clamp for safety against floating error
        if score > 1.0:
            score = 1.0
        elif score < -1.0:
            score = -1.0
        return score


# Example usage (for reference):
# from arena import play_game
# from human_cli_agent import HumanCLI
# if __name__ == "__main__":
#     agent = EvaluationAgent(depth=2, weights=[1.0, 0.5])
#     res = play_game(white=HumanCLI(), black=agent, time_control=None)
#     print(res["result"], res["termination"])
