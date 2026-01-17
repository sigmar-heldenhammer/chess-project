# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 00:19:35 2026

@author: Judson
"""

from .core import ModularMinimaxAgent
from .evals import MaterialEvaluator
from .terminals import DefaultTerminal
from .leaves import DepthZeroLeaf
from .ordering import ActivityOrdering, HistoryOrdering
from .tt import NoTT, SimpleDictTT
from .depth import NoDepthAdjustment
    
# ----------------------------
# Example “preset agent” constructors (optional)
# ----------------------------

def make_basic_minimax(depth: int = 3) -> ModularMinimaxAgent:
    """A simple baseline minimax agent with activity ordering and no TT."""
    return ModularMinimaxAgent(
        depth=depth,
        evaluator=MaterialEvaluator(draw_contempt=0.0, near_rep_nudge=0.0),
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=DepthZeroLeaf(),
        ordering=ActivityOrdering(),
        tt=NoTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=False,
        name=f"BasicMinimax(d={depth})",
    )


def make_tt_minimax(depth: int = 3) -> ModularMinimaxAgent:
    """A TT-enabled minimax agent (simple dict TT)."""
    return ModularMinimaxAgent(
        depth=depth,
        evaluator=MaterialEvaluator(draw_contempt=0.0, near_rep_nudge=0.0),
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=DepthZeroLeaf(),
        ordering=ActivityOrdering(),
        tt=SimpleDictTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=False,
        name=f"TTMinimax(d={depth})",
    )

def make_history_minimax(depth: int = 3) -> ModularMinimaxAgent:
    """A TT-enabled minimax agent (simple dict TT)."""
    return ModularMinimaxAgent(
        depth=depth,
        evaluator=MaterialEvaluator(draw_contempt=0.0, near_rep_nudge=0.0),
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=DepthZeroLeaf(),
        ordering=HistoryOrdering(),
        tt=SimpleDictTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=False,
        name=f"TTMinimax(d={depth})",
    )
