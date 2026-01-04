# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:32:06 2025

@author: Judson
"""

from greedy_material_agent import GreedyMaterialAgent
from minimax_agent import MinimaxAgent
from evaluation_agent_decorator import EvaluationAgent
from quiescence_agent import QuiescenceAgent
from match import play_match  # if you already have it
from divergence_logger import compare_agents_at_root
from random_agent import RandomAgent
from tt_agent import TTAgent
from history_agent import HistoryAgent

def divergence_probe(board, white, black):
    # Always compare both sides' agents from the root; you can also restrict to a specific class
    compare_agents_at_root(board, white, black, out_dir="divergences", sample_rate=0.25)

A = TTAgent(depth=3)
B = HistoryAgent(depth=3)
#B = EvaluationAgent(depth=3, weights={"material_share": 1.0, "center_control": 0.1, "activity": 0.0, "active_pieces": 0.0, "pseudo_active_pieces": 0.1})
#B = QuiescenceAgent(depth=4, use_alpha_beta=True, log_quiescence_diffs=False)
#play_match(A, B, games=1, tc=(600, 0), divergence_probe = divergence_probe)
play_match(A, B, games=1, tc=(1200, 0))

