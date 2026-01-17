# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:32:06 2025

@author: Judson
"""
from match import play_match  # if you already have it



from template_agent import make_tt_minimax
from template_agent import make_basic_minimax
from template_agent import make_history_minimax
# from modular_agent import make_basic_minimax




A = make_basic_minimax(depth=3)
B = make_history_minimax(depth=3)
#B = EvaluationAgent(depth=3, weights={"material_share": 1.0, "center_control": 0.1, "activity": 0.0, "active_pieces": 0.0, "pseudo_active_pieces": 0.1})
#B = QuiescenceAgent(depth=4, use_alpha_beta=True, log_quiescence_diffs=False)
#play_match(A, B, games=1, tc=(600, 0), divergence_probe = divergence_probe)
play_match(A, B, games=5, tc=None)

