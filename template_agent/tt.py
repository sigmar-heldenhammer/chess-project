# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:47:59 2026

@author: Judson
"""


import chess
from .agent_templates import TTProbe, TTFlag
from typing import Optional, Dict, Tuple

class NoTT:
    
    def size(self):
        return -1
    
    def probe(self, board: chess.Board, *, depth: int, alpha: float, beta: float) -> TTProbe:
        return TTProbe(hit=False)

    def store(
        self,
        board: chess.Board,
        *,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None:
        return


class SimpleDictTT:
    """
    A minimal TT that can be swapped in later.
    Stores (depth, value, flag, best_move) keyed by polyglot Zobrist hash.

    NOTE: This does not solve repetition/history draw issues; terminal checks must occur before probe.
    """
    def __init__(self):
        self._table: Dict[int, Tuple[int, float, TTFlag, Optional[chess.Move]]] = {}

    def _key(self, board: chess.Board) -> int:
        # polyglot hash is in python-chess
        return chess.polyglot.zobrist_hash(board)
    
    def size(self) -> int:
        return len(self._table)

    def probe(self, board: chess.Board, *, depth: int, alpha: float, beta: float) -> TTProbe:
        k = self._key(board)
        ent = self._table.get(k)
        if ent is None:
            return TTProbe(hit=False)

        stored_depth, value, flag, best_move = ent
        if stored_depth < depth:
            return TTProbe(hit=False, best_move_hint=best_move, stored_depth=stored_depth)

        # For now, we allow an EXACT hit to return a value.
        # Bounds can be used to tighten alpha/beta (engine-side) if you enable that later.
        return TTProbe(
            hit=True,
            value=value,
            flag=flag,
            best_move_hint=best_move,
            stored_depth=stored_depth,
        )

    def store(
        self,
        board: chess.Board,
        *,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None:
        k = self._key(board)
        prev = self._table.get(k)
        if prev is None or depth >= prev[0]:
            self._table[k] = (depth, value, flag, best_move)

