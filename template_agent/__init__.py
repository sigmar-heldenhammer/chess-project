# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 00:18:08 2026

@author: Judson
"""

from .core import ModularMinimaxAgent

from .presets import (
    make_basic_minimax,
    make_tt_minimax,
    make_history_minimax,   # once you add it
)

__all__ = [
    "ModularMinimaxAgent",
    "make_basic_minimax",
    "make_tt_minimax",
    "make_history_minimax",
]
