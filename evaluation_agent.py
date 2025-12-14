"""
evaluation_agent.py

An agent that extends MinimaxAgent and overrides _evaluate to use a weighted
average of simple criteria in the range [-1, 1], while preserving terminal
handling (mate/draw) from the parent.

Criteria implemented:
  1) Material share ratio in [-1, 1]:
     ((friendly_material) / (total_material)) mapped from [0,1] to [-1,1]
     via: 2 * (friendly / total) - 1. (Kings ignored as value 0.)

  2) Central control in [-1, 1]:
     ((#friendly in {c3..f6}) - (#hostile in {c3..f6})) / 16

  3) Activity (mobility) in [-1, 1]:
     (legal_moves_friendly - legal_moves_opponent) /
     max(1, legal_moves_friendly + legal_moves_opponent)
"""

from __future__ import annotations

import chess
from typing import List, Optional, Sequence, Tuple

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
        ordering_depth: int = 1
    ) -> None:
        super().__init__(depth=depth, seed=seed, use_alpha_beta=use_alpha_beta, order_moves=order_moves)
        # Default equal weights for three criteria (material, center, activity)
        self.weights: List[float] = list(weights) if weights is not None else [1.0, 1.0, 1.0]
        self._center = tuple(_central_squares())
        self.ordering_depth = max(ordering_depth, 0)

    # --------------- Criteria (each returns value in [-1, 1]) ---------------

    def _crit_material_share(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Material share mapped to [-1, 1]:
            share = friendly_value / total_value in [0,1]
            return 2*share - 1 to get [-1, 1].

        Kings are assigned value 0 in PIECE_VALUES and do not affect totals.
        """
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

    def _crit_activity(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Activity (mobility) normalized to [-1, 1] from the root_color perspective:
            (legal_friendly - legal_opponent) / max(1, legal_friendly + legal_opponent)
        """
        def legal_count_for(color: chess.Color) -> int:
            # Count legal moves for 'color' by toggling side-to-move with a null move when needed.
            if board.turn == color:
                return board.legal_moves.count()
            # Make it 'color' to move without changing the position:
            board.push(chess.Move.null())
            try:
                return board.legal_moves.count()
            finally:
                board.pop()

        friendly_moves = legal_count_for(root_color)
        opponent_moves = legal_count_for(not root_color)
        denom = max(1, friendly_moves + opponent_moves)
        return (friendly_moves - opponent_moves) / float(denom)

    # --------------- Weighted combination + terminal handling ---------------

    def _evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Preserve terminal handling (mate/draw) like the parent, then use
        the weighted blend of criteria (each in [-1,1]).
        """
        # Terminal outcomes first
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
        c3 = self._crit_activity(board, root_color)         # [-1,1]

        crits = (c1, c2, c3)
        weights = self.weights

        # Pad or trim weights to match number of criteria
        if len(weights) < len(crits):
            weights = list(weights) + [1.0] * (len(crits) - len(weights))
        elif len(weights) > len(crits):
            weights = list(weights[:len(crits)])

        # Weighted average, normalize by sum of abs weights to keep final in [-1,1]
        denom = sum(abs(w) for w in weights) or 1.0
        score = sum(w * v for w, v in zip(weights, crits)) / denom

        # Clamp against tiny floating noise
        if score > 1.0:
            score = 1.0
        elif score < -1.0:
            score = -1.0
        return score
    
    def _ordered_moves(self, board: chess.Board, moves) -> list[chess.Move]:
        rc = getattr(self, "_eval_root_color", board.turn)  # fallback: side to move
        root_d = getattr(self, "_ordering_root_depth", 0)
        now_d  = getattr(self, "_ordering_depth_now", 0)
        ply_from_root = root_d - now_d

        if ply_from_root < self.ordering_depth:
            scored = []
            for mv in moves:
                board.push(mv)
                try:
                    # your richer eval from root POV drives ordering
                    s = self._evaluate(board, rc)
                finally:
                    board.pop()
                scored.append((s, mv))
            scored.sort(key=lambda t: t[0], reverse=True)
            return [mv for _, mv in scored]
        else:
            return super()._ordered_moves(board, moves)


# Example usage (for reference):
# from arena import play_game
# from human_cli_agent import HumanCLI
# if __name__ == "__main__":
#     agent = EvaluationAgent(depth=2, weights=[1.0, 1.0, 1.0])
#     res = play_game(white=HumanCLI(), black=agent, time_control=None)
#     print(res["result"], res["termination"])
