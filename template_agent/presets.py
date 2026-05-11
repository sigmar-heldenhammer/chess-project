# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 00:19:35 2026

@author: Judson
"""

from .core import ModularMinimaxAgent
from .evals import MaterialEvaluator, WeightedCriteriaEvaluator
from .terminals import DefaultTerminal
from .leaves import DepthZeroLeaf, NegativeDepthLeaf
from .ordering import ActivityOrdering, HistoryOrdering, QuiescentOrdering
from .tt import NoTT, SimpleDictTT
from .depth import NoDepthAdjustment
from .search_drivers import IterativeDeepeningDriver
    
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
        randomize_ties=True,
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
        randomize_ties=True,
        name=f"TTMinimax(d={depth})",
    )

def make_history_minimax(depth: int = 3) -> ModularMinimaxAgent:
    """A TT-enabled minimax agent that adds a history heuristic (simple dict TT)."""
    return ModularMinimaxAgent(
        depth=depth,
        evaluator=MaterialEvaluator(draw_contempt=0.0, near_rep_nudge=0.0),
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=DepthZeroLeaf(),
        ordering=HistoryOrdering(),
        tt=SimpleDictTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=True,
        name=f"HHMinimax(d={depth})",
    )


def make_id_minimax(depth: int = 3) -> ModularMinimaxAgent:
    """A minimax agent with iterative deepening, transposition table, and history heuristic"""
    return ModularMinimaxAgent(
        depth=depth,
        driver=IterativeDeepeningDriver(),
        evaluator=MaterialEvaluator(draw_contempt=0.0, near_rep_nudge=0.0),
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=DepthZeroLeaf(),
        ordering=HistoryOrdering(),
        tt=SimpleDictTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=True,
        name=f"IDMinimax(d={depth})",
    )

def make_eval_minimax(depth: int = 3) -> ModularMinimaxAgent:
    return ModularMinimaxAgent(
        depth=depth,
        evaluator = WeightedCriteriaEvaluator(
            {
                "material_balance": 1.0,
                "center_control": 1.0,
                "pseudo_active_pieces": 1.0,
                "pawn_structure": 1.0,
            },
            normalize=False,
            clamp=False
        )
        ,
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=DepthZeroLeaf(),
        ordering=ActivityOrdering(),
        tt=NoTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=True,
        name=f"EvalMinimax(d={depth})",
        )



def make_qsearch_agent(depth: int = 3, min_depth: int = -3) -> ModularMinimaxAgent:
    return ModularMinimaxAgent(
        depth=depth,
        evaluator = WeightedCriteriaEvaluator(
            {
                "material_balance": 1.0,
                "center_control": 1.0,
                "pseudo_active_pieces": 1.0,
                "pawn_structure": 1.0,
            },
            normalize=False,
            clamp=False
        )
        ,
        terminal=DefaultTerminal(draw_contempt=0.0),
        leaf=NegativeDepthLeaf(min_depth=min_depth),
        ordering=QuiescentOrdering(
            base_ordering=ActivityOrdering(),
            q_start_depth=0,
            include_evasions=True,
        ),
        tt=NoTT(),
        depth_policy=NoDepthAdjustment(),
        randomize_ties=True,
        name=f"QSearchMinimax(d={depth})",
        )