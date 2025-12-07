# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:32:06 2025

@author: Judson
"""

from greedy_material_agent import GreedyMaterialAgent
from minimax_agent import MinimaxAgent
from evaluation_agent import EvaluationAgent
from quiescence_agent import QuiescenceAgent
from match import play_match  # if you already have it

#A = EvaluationAgent(depth=2, seed=3, use_alpha_beta=False, weights=[1.0, 0.1, 0.1])
A = MinimaxAgent(depth=3, use_alpha_beta=True)
B = QuiescenceAgent(depth=3, use_alpha_beta=True, log_quiescence_diffs=True, log_limit=99, log_threshold=4.5)
play_match(A, B, games=1, tc=(600, 0))
