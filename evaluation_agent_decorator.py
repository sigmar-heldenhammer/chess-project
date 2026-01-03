"""
evaluation_agent.py

An agent that extends MinimaxAgent and overrides _evaluate to use a weighted
average of simple criteria, while preserving terminal handling (mate/draw)
from the parent.

This version uses a decorator-based "criteria registry" (Pattern B):
- Each criterion method is annotated with @criterion(...)
- Criteria and their default weights are auto-discovered at init time
- Any criterion with weight == 0.0 is NOT evaluated at all

Compatibility:
- `weights` may be passed as a positional Sequence[float] (legacy behavior)
  and will be padded/trimmed to match the number of registered criteria,
  in a stable order controlled by each criterion's `order`.
- Optionally, `weights` may also be passed as a dict {criterion_name: weight}.
"""

from __future__ import annotations

import chess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

# We extend the existing MinimaxAgent from minimax_agent.py
from minimax_agent import MinimaxAgent, PIECE_VALUES, MATE_VALUE


# ---------------------- Criterion decorator (Pattern B) ----------------------

def criterion(name: str, default_weight: float = 1.0, order: int = 0):
    """
    Decorator to mark a method as an evaluation criterion.

    The decorator *does not* wrap the function; it simply attaches metadata
    used to auto-build a registry at runtime.

    Args:
        name: External key for the criterion (used in weights dict).
        default_weight: Default weight if not provided by the caller.
        order: Controls criterion ordering (for legacy positional weights).
               Lower numbers come first.
    """
    def _decorator(fn: Callable[..., float]) -> Callable[..., float]:
        setattr(fn, "_criterion_name", name)
        setattr(fn, "_criterion_default_weight", float(default_weight))
        setattr(fn, "_criterion_order", int(order))
        return fn
    return _decorator


def _central_squares() -> List[int]:
    """Squares for the 4x4 center c3..f6 (files c-f, ranks 3-6)."""
    files = ["c", "d", "e", "f"]
    ranks = ["3", "4", "5", "6"]
    return [chess.parse_square(f + r) for f in files for r in ranks]


class EvaluationAgent(MinimaxAgent):
    """
    Minimax agent with a custom evaluation function that blends multiple
    (nominally) normalized criteria via weights.

    Criteria are registered via the @criterion decorator.
    """

    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        weights: Optional[Union[Sequence[float], Dict[str, float]]] = None,
        ordering_depth: int = 1,
        draw_contempt: float = 0.2
    ) -> None:
        super().__init__(depth=depth, seed=seed, use_alpha_beta=use_alpha_beta, order_moves=order_moves)

        self._center = tuple(_central_squares())
        self.ordering_depth = max(ordering_depth, 0)
        self.draw_contempt = float(draw_contempt)


        # Discover criteria (methods annotated with @criterion)
        self._criteria: List[Tuple[str, Callable[[chess.Board, chess.Color], float], float, int]] = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr):
                continue
            if hasattr(attr, "_criterion_name"):
                name = getattr(attr, "_criterion_name")
                default_w = float(getattr(attr, "_criterion_default_weight", 0.0))
                order = int(getattr(attr, "_criterion_order", 0))
                # attr is already a bound method: (board, root_color) -> float
                self._criteria.append((name, attr, default_w, order))

        # Stable ordering: primarily by `order`, then by name
        self._criteria.sort(key=lambda t: (t[3], t[0]))

        # Initialize weights dict from defaults
        self.weights_by_name: Dict[str, float] = {name: default_w for name, _, default_w, _ in self._criteria}

        # Legacy positional weights list (kept for compatibility/debugging)
        self.weights: List[float] = [self.weights_by_name[name] for name, _, _, _ in self._criteria]

        # Apply caller-provided weights
        if weights is not None:
            if isinstance(weights, dict):
                # Dict weights are the preferred flexible format
                for k, v in weights.items():
                    if k in self.weights_by_name:
                        self.weights_by_name[k] = float(v)
                # Refresh positional list to match registry order
                self.weights = [self.weights_by_name[name] for name, _, _, _ in self._criteria]
            else:
                # Positional weights: preserve old pad/trim behavior against criterion order.
                seq = list(weights)
                if len(seq) < len(self._criteria):
                    seq = seq + [1.0] * (len(self._criteria) - len(seq))
                elif len(seq) > len(self._criteria):
                    seq = seq[:len(self._criteria)]
                self.weights = [float(x) for x in seq]
                # Also populate the dict for the skip-when-zero behavior
                for (name, _, _, _), w in zip(self._criteria, self.weights):
                    self.weights_by_name[name] = float(w)

    # --------------- Criteria (each returns a float, intended to be in [-1, 1]) ---------------

    @criterion("material_share", default_weight=1.0, order=10)
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
        return 2.0 * share - 1.0  # map to [-1,1]

    @criterion("center_control", default_weight=1.0, order=20)
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

    @criterion("activity", default_weight=1.0, order=30)
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

    @criterion("active_pieces", default_weight=1.0, order=40)
    def _crit_active_pieces(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Number of active pieces for root_color relative to total number on the board,
        normalized on [-1, -1] (legacy docstring preserved).

        Active pieces defined as those that have three or more legal moves.
        (Legacy behavior preserved.)
        """
        def active_piece_count_for(color: chess.Color) -> int:
            active_pieces = 0
            prev_turn = board.turn
            try:
                # Legacy behavior: directly override board.turn
                board.turn = color

                # Count legal moves by from-square
                counts_by_square: Dict[chess.Square, int] = defaultdict(int)
                for mv in board.legal_moves:
                    counts_by_square[mv.from_square] += 1

                for sq, piece in board.piece_map().items():
                    if piece.color == color and counts_by_square.get(sq, 0) >= 3:
                        active_pieces += 1

                return active_pieces
            finally:
                board.turn = prev_turn

        active_pieces = active_piece_count_for(root_color)
        if board.turn == root_color:
            return float(active_pieces)
        else:
            return float(-1 * active_pieces)
        
    @criterion("pseudo_active_pieces", default_weight=1.0, order=50)
    def _crit_pseudo_active_pieces(self, board: chess.Board, root_color: chess.Color) -> float:
        occ_own = board.occupied_co[root_color]
        active = 0.0
        for sq, p in board.piece_map().items():
            if p.color != root_color:
                continue
            else:
                if len((board.attacks(sq) & ~occ_own)) >= 3:
                    active += 1
        active = active / 8.0
        if board.turn == root_color:
            return float(active)
        else:
            return float(-1*active)
        


    # --------------- Weighted combination + terminal handling ---------------

    def _evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Preserve terminal handling (mate/draw) like the parent, then use
        the weighted blend of criteria.

        New behavior:
          - Criteria with weight == 0.0 are skipped (not evaluated at all).
          - Weighted average normalized by sum(abs(weights)) over evaluated criteria.
        """
        # Terminal outcomes first


        # Weighted average, normalize by sum of abs weights.
        total = 0.0
        denom = 0.0

        # Iterate criteria in stable order
        for (name, fn, _default_w, _order), w_pos in zip(self._criteria, self.weights):
            # Prefer dict weight (allows dict updates post-init), fallback to positional
            w = float(self.weights_by_name.get(name, w_pos))
            if w == 0.0:
                continue  # <-- do not compute criterion at all
            v = float(fn(board, root_color))
            total += w * v
            denom += abs(w)

        score = total / (denom or 1.0)

        # Clamp against tiny floating noise
        if score > 1.0:
            score = 1.0
        elif score < -1.0:
            score = -1.0
            
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
            if res == "1-0":
                return MATE_VALUE if root_color == chess.WHITE else -MATE_VALUE
            if res == "0-1":
                return MATE_VALUE if root_color == chess.BLACK else -MATE_VALUE
            return 0.0  # draw
        
            c   = self.draw_contempt
            return (-c) if score >= 0.0 else (+c)

        try:
            if board.is_repetition(2) or board.can_claim_threefold_repetition():
                #print("draw warning, score undefined")
                c = 0.5 * self.draw_contempt  # smaller nudge than terminal draw
                #print("c " + c)
                if score >= 0.0:
                    score -= c   # if we're better, discourage repeating
                else:
                    score += c   # if we're worse, repeating is fine (can aim for a draw)
                #print(f"draw warning {score}")
        except Exception:
            pass
        
        return score

    def _ordered_moves(self, board: chess.Board, moves) -> List[chess.Move]:
        rc = getattr(self, "_eval_root_color", board.turn)  # fallback: side to move
        root_d = getattr(self, "_ordering_root_depth", 0)
        now_d = getattr(self, "_ordering_depth_now", 0)
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
#     # legacy positional:
#     agent = EvaluationAgent(depth=2, weights=[1.0, 1.0, 0.0, 1.0])
#     # dict-based:
#     # agent = EvaluationAgent(depth=2, weights={"activity": 0.0, "active_pieces": 0.0})
#     res = play_game(white=HumanCLI(), black=agent, time_control=None)
#     print(res["result"], res["termination"])
