# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:32:06 2025

@author: Judson
"""

from greedy_material_agent import GreedyMaterialAgent
from minimax_agent import MinimaxAgent
from evaluation_agent import EvaluationAgent
from match import play_match  # if you already have it

A = EvaluationAgent(depth=3, seed=1, use_alpha_beta=False, weights=[1.0, 1.0, 1.0])
B = MinimaxAgent(depth=3, seed=2, use_alpha_beta=False)
play_match(A, B, games=5, tc=(600, 0))
