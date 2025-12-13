
"""
quiescence_agent.py

A MinimaxAgent extension that adds a quiescence search to reduce horizon effects.
At depth == 0 in the main search, instead of returning a static evaluation,
we call _quiescent_search(), which:
  - explores only CAPTURE moves (no quiet moves)
  - uses alpha-beta within the quiescent search
  - terminates when there are no legal captures, then returns a static eval

Logging (for manual investigation):
  - When enabled, records cases where |qscore - stand_pat| >= log_threshold.
  - Each record is written as a single JSON file into log_dir.
  - Contains entry FEN, after-best-move FEN, root_color, side-to-move, maximizing,
    stand_pat, qscore, delta, in_check, best_first_move (UCI), and simple counts.
  - Only the first `log_limit` qualifying cases are logged (per agent instance).

Default log_threshold is 2.5 (≈ a full piece), per your preference.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
import chess
from typing import Optional

from minimax_agent import MinimaxAgent, PIECE_VALUES, MATE_VALUE


class QuiescenceAgent(MinimaxAgent):
    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        *,
        # Logging controls
        log_quiescence_diffs: bool = False,
        log_dir: str = "quiescence_diffs",
        log_threshold: float = 2.5,   # ≈ a full piece
        log_limit: int = 10,
    ) -> None:
        super().__init__(depth=depth, seed=seed, use_alpha_beta=use_alpha_beta, order_moves=order_moves)
        # Logging config
        self.log_quiescence_diffs = bool(log_quiescence_diffs)
        self.log_dir = log_dir
        self.log_threshold = float(log_threshold)
        self.log_limit = int(log_limit)
        self._qdiff_count = 0  # number of records written so far

    # ----------------- overrides -----------------

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
        PV-aware alpha-beta: returns (score, pv), where pv is a list of chess.Move.
        Leaf returns (eval, []).
        """
 
        # Terminal (mate/draw/claim)
        if board.is_game_over(claim_draw=True):
            return self._evaluate(board, root_color), []
    
        # Depth cutoff: static evaluation (or overridden by subclasses)
        if depth == 0:
            return self._quiescent_search(board, root_color, maximizing, alpha, beta), []
    
    
        legal = list(board.legal_moves)
        if not legal:
            return self._evaluate(board, root_color), []
    
        if self.order_moves:
            legal = self._ordered_moves(board, legal)
    

    
        if maximizing:
            best_val = -math.inf
            best_pv: list[chess.Move] = []
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
                        break  # beta cutoff
            return best_val, best_pv
        else:
            best_val = math.inf
            best_pv: list[chess.Move] = []
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
                        break  # alpha cutoff
            return best_val, best_pv

    # ----------------- quiescence -----------------

    def _quiescent_search(
        self,
        board: chess.Board,
        root_color: chess.Color,
        maximizing: bool,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Alpha-beta quiescence:
        - stand-pat evaluation first
        - search only capturing moves
        - stop when no captures exist

        NOTE: This is the "captures-only" version for investigation; logging is added
        to compare against the stand-pat (what plain minimax would have returned).
        """
        # Record entry context for logging
        entry_fen = board.fen()
        entry_turn = board.turn  # True=White, False=Black
        in_check = board.is_check()

        # Stand-pat: what plain minimax would return at this leaf
        stand_pat = self._evaluate(board, root_color)
        
        if board.is_check():
            active_moves = [m for m in board.legal_moves]

        else:
        # Apply stand-pat bounds
            if maximizing:
                if self.use_alpha_beta and stand_pat >= beta:
                    qscore = stand_pat
                    self._maybe_log_qdiff(entry_fen, root_color, entry_turn, maximizing, in_check,
                                          stand_pat, qscore, best_first_move=None, captures_considered=0, captures_searched=0)
                    return qscore
                if stand_pat > alpha:
                    alpha = stand_pat
            else:
                if self.use_alpha_beta and stand_pat <= alpha:
                    qscore = stand_pat
                    self._maybe_log_qdiff(entry_fen, root_color, entry_turn, maximizing, in_check,
                                          stand_pat, qscore, best_first_move=None, captures_considered=0, captures_searched=0)
                    return qscore
                if stand_pat < beta:
                    beta = stand_pat
    
            # Generate capture moves only
            active_moves = [m for m in board.legal_moves if (board.is_capture(m) or m.promotion)]

        if not active_moves:
            qscore = stand_pat
            self._maybe_log_qdiff(entry_fen, root_color, entry_turn, maximizing, in_check,
                                  stand_pat, qscore, best_first_move=None, captures_considered=0, captures_searched=0)
            return qscore

        # Simple capture ordering (MVV-LVA-ish) to improve pruning
        def cap_score(mv: chess.Move) -> float:
            victim = board.piece_at(mv.to_square)
            attacker = board.piece_at(mv.from_square)
            v = PIECE_VALUES.get(victim.piece_type, 0.0) if victim else 0.0
            a = PIECE_VALUES.get(attacker.piece_type, 0.0) if attacker else 0.0
            promo = PIECE_VALUES.get(mv.promotion, 0.0) if mv.promotion else 0.0
            return 10.0 * v - 0.1 * a + promo

        active_moves.sort(key=cap_score, reverse=True)

        best_first_move = None
        captures_searched = 0

        if maximizing:
            value = stand_pat
            for mv in active_moves:
                board.push(mv)
                child = self._quiescent_search(board, root_color, False, alpha, beta)
                board.pop()
                captures_searched += 1
                if child > value:
                    value = child
                    best_first_move = mv
                if self.use_alpha_beta:
                    if value > alpha:
                        alpha = value
                    if alpha >= beta:
                        break  # cutoff
            qscore = value
        else:
            value = stand_pat
            for mv in active_moves:
                board.push(mv)
                child = self._quiescent_search(board, root_color, True, alpha, beta)
                board.pop()
                captures_searched += 1
                if child < value:
                    value = child
                    best_first_move = mv
                if self.use_alpha_beta:
                    if value < beta:
                        beta = value
                    if alpha >= beta:
                        break  # cutoff
            qscore = value

        # Log if difference is significant
        self._maybe_log_qdiff(
            entry_fen, root_color, entry_turn, maximizing, in_check,
            stand_pat, qscore, best_first_move=best_first_move,
            captures_considered=len(active_moves),
            captures_searched=captures_searched,
        )
        return qscore

    # ----------------- logging helpers -----------------

    def _maybe_log_qdiff(
        self,
        entry_fen: str,
        root_color: chess.Color,
        entry_turn: bool,
        maximizing: bool,
        in_check: bool,
        stand_pat: float,
        qscore: float,
        *,
        best_first_move: Optional[chess.Move],
        captures_considered: int,
        captures_searched: int,
    ) -> None:
        """Write a single JSON record if abs difference exceeds threshold and under limit."""
        if not self.log_quiescence_diffs:
            return
        if self._qdiff_count >= self.log_limit:
            return

        delta = abs(qscore - stand_pat)
        if delta < self.log_threshold:
            return

        payload = {
            "entry_fen": entry_fen,
            "after_best_move_fen": None,
            "root_color": "white" if root_color == chess.WHITE else "black",
            "side_to_move_at_entry": "white" if entry_turn else "black",
            "maximizing_at_entry": bool(maximizing),
            "in_check_at_entry": bool(in_check),
            "stand_pat": stand_pat,
            "qscore": qscore,
            "delta": delta,
            "best_first_move_uci": best_first_move.uci() if best_first_move else None,
            "captures_considered": int(captures_considered),
            "captures_searched": int(captures_searched),
        }

        if best_first_move is not None:
            tmp = chess.Board(entry_fen)
            try:
                tmp.push(best_first_move)
                payload["after_best_move_fen"] = tmp.fen()
            except Exception:
                payload["after_best_move_fen"] = None

        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._qdiff_count += 1
            fname = f"qdiff_{self._qdiff_count:03d}.json"
            fpath = Path(self.log_dir) / fname
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[quiescence] Failed to write log file: {e}")

