"""
evaluation_agent.py

An agent that extends MinimaxAgent and overrides _evaluate.

Two evaluation modes:
1) Legacy weighted-criteria mode (weights=[...]) using criteria in [-1, 1].
2) Model-driven mode (model_path="...json") loading a logistic-regression model exported from your
   stepwise workflow. In this mode, _evaluate returns a win-probability for root_color in [0, 1]
   (mates/draws still handled with +/-MATE_VALUE / 0.0).

Model file format (JSON):
{
  "version": 1,
  "model_type": "logistic_regression",
  "target": "stm_wins",
  "features": ["material_share", "center_control", "activity"],
  "intercept": -0.123,
  "coefficients": {
     "material_share": 0.9,
     "center_control": 0.2,
     "activity": 0.1
  }
}

Important:
- The model is assumed to predict P(side_to_move wins) = sigmoid(intercept + sum_i coef_i * x_i),
  where x_i are computed from the SIDE-TO-MOVE perspective.
- _evaluate converts this to P(root_color wins) by inverting when board.turn != root_color:
      p_root = p_stm if board.turn == root_color else (1 - p_stm)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import chess

# We extend the existing MinimaxAgent from minimax_agent.py
from minimax_agent import MinimaxAgent, PIECE_VALUES, MATE_VALUE


def _central_squares() -> list[int]:
    """Squares for the 4x4 center c3..f6 (files c-f, ranks 3-6)."""
    files = ['c', 'd', 'e', 'f']
    ranks = ['3', '4', '5', '6']
    return [chess.parse_square(f + r) for f in files for r in ranks]


def _sigmoid(z: float) -> float:
    # numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class EvaluationAgent(MinimaxAgent):
    """
    Minimax agent with a custom evaluation function.
    """

    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        # Legacy weighted-criteria mode:
        weights: Optional[Sequence[float]] = None,
        ordering_depth: int = 1,
        # Model-driven mode:
        model_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(depth=depth, seed=seed, use_alpha_beta=use_alpha_beta, order_moves=order_moves)

        self._center = tuple(_central_squares())
        self.ordering_depth = max(ordering_depth, 0)

        # Legacy weights mode (default)
        self.weights: List[float] = list(weights) if weights is not None else [1.0, 1.0, 1.0]

        # Model mode (optional)
        self.model_path: Optional[Path] = Path(model_path) if model_path is not None else None
        self._model: Optional[dict] = None
        if self.model_path is not None:
            self._model = self._load_and_validate_model(self.model_path)

    # -----------------
    # Criteria registry
    # -----------------

    def _crit_material_share(self, board: chess.Board, pov: chess.Color) -> float:
        """
        Material share mapped to [-1, 1]:
            share = friendly_value / total_value in [0,1]
            return 2*share - 1 to get [-1, 1].
        """
        def side_value(color: chess.Color) -> float:
            s = 0.0
            for ptype, val in PIECE_VALUES.items():
                if val == 0.0:
                    continue
                s += val * len(board.pieces(ptype, color))
            return s

        friendly = side_value(pov)
        hostile = side_value(not pov)
        total = friendly + hostile
        if total <= 0.0:
            return 0.0  # only kings; neutral
        share = friendly / total  # 0..1
        return 2.0 * share - 1.0

    def _crit_center_control(self, board: chess.Board, pov: chess.Color) -> float:
        """
        (#friendly in c3..f6 - #hostile in c3..f6) / 16 in [-1, 1].
        """
        friendly = 0
        hostile = 0
        for sq in self._center:
            piece = board.piece_at(sq)
            if not piece:
                continue
            if piece.color == pov:
                friendly += 1
            else:
                hostile += 1
        return (friendly - hostile) / 16.0

    def _crit_activity(self, board: chess.Board, pov: chess.Color) -> float:
        """
        Activity (mobility) normalized to [-1, 1] from the pov perspective:
            (legal_pov - legal_opp) / max(1, legal_pov + legal_opp)
        """
        def legal_count_for(color: chess.Color) -> int:
            if board.turn == color:
                return board.legal_moves.count()
            board.push(chess.Move.null())
            try:
                return board.legal_moves.count()
            finally:
                board.pop()

        pov_moves = legal_count_for(pov)
        opp_moves = legal_count_for(not pov)
        denom =  pov_moves + opp_moves
        if denom <= 0:
            return 0.0
        return (pov_moves - opp_moves) / float(denom)

    # Additional feature candidates (to match common dataset columns)
    def _feat_pieces_total(self, board: chess.Board, pov: chess.Color) -> float:
        """Total pieces on board (both sides), excluding kings."""
        total = 0
        for ptype, val in PIECE_VALUES.items():
            if val == 0.0:
                continue
            total += len(board.pieces(ptype, chess.WHITE))
            total += len(board.pieces(ptype, chess.BLACK))
        return float(total)

    def _feat_material_fraction(self, board: chess.Board, pov: chess.Color) -> float:
        """Material fraction in [0,1] from pov: (material_share + 1)/2."""
        return (self._crit_material_share(board, pov) + 1.0) / 2.0

    def _feat_center_occupancy(self, board: chess.Board, pov: chess.Color) -> float:
        """Center occupancy in [0,1] from pov: (#pov pieces in center)/16."""
        friendly = 0
        for sq in self._center:
            piece = board.piece_at(sq)
            if piece and piece.color == pov:
                friendly += 1
        return friendly / 16.0

    def _feat_mobility_ratio(self, board: chess.Board, pov: chess.Color) -> float:
        """Mobility ratio in [0,1]: legal_pov / max(1, legal_pov + legal_opp)."""
        def legal_count_for(color: chess.Color) -> int:
            if board.turn == color:
                return board.legal_moves.count()
            board.push(chess.Move.null())
            try:
                return board.legal_moves.count()
            finally:
                board.pop()

        pov_moves = legal_count_for(pov)
        opp_moves = legal_count_for(not pov)
        denom = max(1, pov_moves + opp_moves)
        return pov_moves / float(denom)
    
    def _feat_is_check_stm(self, board: chess.Board, pov: chess.Color) -> float:
        return 1 if board.is_check() else 0

    def _feature_registry(self) -> Dict[str, Callable[[chess.Board, chess.Color], float]]:
        """
        Map feature names (as stored in model JSON) to functions that compute them.
        These must be cheap because they're called in the minimax inner loop.
        """
        return {
            # canonical names aligned with current criteria
            "material_share": self._crit_material_share,
            "center_control": self._crit_center_control,
            "activity": self._crit_activity,

            # aliases that commonly appear in your parquet workflow
            "material_fraction": self._crit_material_share,
            "center_occupancy": self._feat_center_occupancy,
            "mobility_ratio": self._crit_activity,
            "pieces_total": self._feat_pieces_total,
            "is_check_stm": self._feat_is_check_stm,
            
            # composites (if you trained them)
            "material_x_pieces": lambda b, c: self._crit_material_share(b, c) * self._feat_pieces_total(b, c),
            "center_x_pieces":   lambda b, c: self._feat_center_occupancy(b, c) * self._feat_pieces_total(b, c),
            "mobility_x_pieces": lambda b, c: self._crit_activity(b, c) * self._feat_pieces_total(b, c),
        }

    # -----------------
    # Model loading
    # -----------------

    def _load_and_validate_model(self, path: Path) -> dict:
        """
        Loads model JSON and validates it against the feature registry.
        Raises ValueError on any format/consistency issue.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise ValueError("Model file must be a JSON object")

        if raw.get("version") != 1:
            raise ValueError(f"Unsupported model version: {raw.get('version')} (expected 1)")

        if raw.get("model_type") != "logistic_regression":
            raise ValueError(f"Unsupported model_type: {raw.get('model_type')}")

        features = raw.get("features")
        coefs = raw.get("coefficients")
        intercept = raw.get("intercept")

        if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
            raise ValueError("Model 'features' must be a list[str]")

        if not isinstance(coefs, dict) or not all(isinstance(k, str) for k in coefs.keys()):
            raise ValueError("Model 'coefficients' must be a dict[str, number]")

        try:
            intercept = float(intercept)
        except Exception as e:
            raise ValueError(f"Model 'intercept' must be numeric: {e}")

        reg = self._feature_registry()

        # Feature list must be supported by registry and have matching coefficients.
        for f in features:
            if f not in reg:
                raise ValueError(f"Model references unsupported feature: {f}")
            if f not in coefs:
                raise ValueError(f"Model missing coefficient for feature: {f}")

        # Coefs must be numeric; allow extra keys but warn by ignoring them.
        cleaned_coefs: Dict[str, float] = {}
        for f in features:
            try:
                cleaned_coefs[f] = float(coefs[f])
            except Exception as e:
                raise ValueError(f"Coefficient for {f!r} must be numeric: {e}")

        return {
            "version": 1,
            "model_type": "logistic_regression",
            "target": raw.get("target", "stm_wins"),
            "features": features,
            "intercept": intercept,
            "coefficients": cleaned_coefs,
        }

    # -----------------
    # Evaluation
    # -----------------

    def _evaluate_with_model(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Uses the loaded logistic regression model to return P(root_color wins) in [0,1].
        """
        assert self._model is not None
        reg = self._feature_registry()

        # IMPORTANT: compute model inputs from side-to-move perspective
        stm_color = board.turn

        z = float(self._model["intercept"])
        for f in self._model["features"]:
            z += float(self._model["coefficients"][f]) * float(reg[f](board, stm_color))

        p_stm = _sigmoid(z)  # P(side_to_move wins)

        # Convert to P(root_color wins)
        return p_stm if stm_color == root_color else (1.0 - p_stm)

    def _evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Preserve terminal handling (mate/draw) like the parent.

        If model_path was provided, return P(root_color wins) in [0,1] for non-terminal positions.
        Otherwise, use the legacy weighted blend of criteria (each in [-1,1]) and return in [-1,1].
        """
        # Terminal outcomes first
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
            if res == "1-0":
                return MATE_VALUE if root_color == chess.WHITE else -MATE_VALUE
            if res == "0-1":
                return MATE_VALUE if root_color == chess.BLACK else -MATE_VALUE
            return 0.0  # draw

        # Model mode
        if self._model is not None:
            return self._evaluate_with_model(board, root_color)

        # Legacy weighted mode
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
        now_d = getattr(self, "_ordering_depth_now", 0)
        ply_from_root = root_d - now_d

        if ply_from_root < self.ordering_depth:
            scored = []
            for mv in moves:
                board.push(mv)
                try:
                    s = self._evaluate(board, rc)
                finally:
                    board.pop()
                scored.append((s, mv))
            scored.sort(key=lambda t: t[0], reverse=True)
            return [mv for _, mv in scored]
        else:
            return super()._ordered_moves(board, moves)
