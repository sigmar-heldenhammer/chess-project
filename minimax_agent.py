# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 21:05:46 2025

@author: Judson
"""

import math
import random
import chess
from typing import Optional, Tuple
from agents import Agent

# Simple material values (kings excluded)
PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   0.0,
}

MATE_VALUE = 1_000_000.0  # large magnitude so mate dominates material

def _material_balance(board: chess.Board, color: chess.Color) -> float:
    """(my material) - (their material)."""
    opp = not color
    score = 0.0
    for p, v in PIECE_VALUES.items():
        if p == chess.KING:
            continue
        score += v * (len(board.pieces(p, color)) - len(board.pieces(p, opp)))
    return score




class MinimaxAgent(Agent):
    """
    Depth-limited minimax with optional alpha-beta pruning and simple move ordering.
    """

    def __init__(
        self,
        depth: int = 2,
        seed: Optional[int] = None,
        use_alpha_beta: bool = True,
        order_moves: bool = True,
        draw_contempt: float = 0.2
    ):
        assert depth >= 1
        self.depth = depth
        self.rand = random.Random(seed)
        self.use_alpha_beta = use_alpha_beta
        self.order_moves = order_moves
        self.draw_contempt = float(draw_contempt)



    def _evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        """
        Return score from root_color's perspective (pawns).
        Adds 'contempt' vs draws to reduce repeat/stalemate loops.
        """
        # 1) Terminal handling (mate / stalemate / repetition / 50-move, etc.)
        if board.is_game_over(claim_draw=True):
            #print("terminal flag- type undefined")
            res = board.result(claim_draw=True)
            #print(f"terminal state flag: {res}")
            if res == "1-0":
                return MATE_VALUE if root_color == chess.WHITE else -MATE_VALUE
            if res == "0-1":
                return MATE_VALUE if root_color == chess.BLACK else -MATE_VALUE
            # Draw: bias with contempt depending on whether we're better or worse
            mat = _material_balance(board, root_color)
            c   = self.draw_contempt

            # If we're better (mat >= 0), we *dislike* a draw; if worse, we *like* a draw.
            return (-c) if mat >= 0.0 else (+c)
    
        # 2) Stand-pat score
        score = _material_balance(board, root_color)
    
        # 3) Near-repetition soft penalty so the engine feels it *before* the draw is terminal
        #    python-chess tracks repetition via its internal stack; count>=2 means "we've been here once already".
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
    
    # ---- public API expected by the arena ----
    def select_move(self, board: chess.Board, *, color: chess.Color | None = None, **kwargs) -> chess.Move:
        """
        Returns the chosen move (unchanged public API).
        Also records:
            self.last_root_best_move
            self.last_root_score
            self.last_root_pv
        """
        root_color = board.turn if color is None else color
        alpha = float("-inf")
        beta  = float("+inf")
        best_move = None
    
        # PV-aware call
        out = self._search(board, self.depth, root_color, maximizing=True, alpha=alpha, beta=beta)
        #print(color, out[0], str(out[1][0]))
    
        # Backward compatibility: _search may return float or (float, pv)
        if isinstance(out, tuple) and len(out) == 2:
            best_score, pv = out
        else:
            best_score, pv = float(out), []
    
        # If we have a PV, first move in PV is the best root move
        if pv:
            best_move = pv[0]
        else:
            # Fallback if some subclass returned only a float: do a minimal “pick any best”
            # (preserves behavior, though normally pv[0] should exist)
            best_val = float("-inf")
            for mv in board.legal_moves:
                board.push(mv)
                child = self._evaluate(board, root_color)
                board.pop()
                if child > best_val:
                    best_val, best_move = child, mv
    
        # Record diagnostics
        self.last_root_best_move = best_move
        self.last_root_score = best_score
        self.last_root_pv = pv
    
        return best_move


    # ---- core search ----
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
            return self._evaluate(board, root_color), []    
    
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


    # ---- simple move ordering heuristics ----
    def _ordered_moves(self, board: chess.Board, moves) -> list[chess.Move]:
        """
        Very light ordering: captures first (MVV-LVA-ish), then promotions,
        then checks, then the rest. This helps alpha-beta a lot.
        """
        scored: list[Tuple[float, chess.Move]] = []
        for mv in moves:
            score = 0.0
            if board.is_capture(mv):
                victim = board.piece_at(mv.to_square)
                attacker = board.piece_at(mv.from_square)
                v = PIECE_VALUES.get(victim.piece_type, 0.0) if victim else 0.0
                a = PIECE_VALUES.get(attacker.piece_type, 0.0) if attacker else 0.0
                score += 10.0 * v - 0.1 * a  # MVV-LVA-ish
            if mv.promotion:
                score += 5.0 + PIECE_VALUES.get(mv.promotion, 0.0)
            # Quick check bonus (requires push/pop to test safely)
            board.push(mv)
            if board.is_check():
                score += 0.5
            board.pop()
            scored.append((score, mv))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [mv for _, mv in scored]
