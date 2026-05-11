# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:46:49 2026

@author: Judson
"""

from .agent_templates import Evaluator, SearchContext, OrderingPolicy
from typing import Optional



class DepthZeroLeaf:
    """Stop expanding only when depth==0; return static evaluator score."""
    def leaf_value(self, ctx: SearchContext, evaluator: Evaluator, ordering: OrderingPolicy) -> Optional[float]:
        if ctx.depth > 0:
            return None
        return evaluator.evaluate(ctx.board, ctx.root_color)



class NegativeDepthLeaf:
    def __init__(self, min_depth=-6):
        self.min_depth = min_depth

    def leaf_value(self, ctx, evaluator, ordering):
        if ctx.depth > self.min_depth:
            return None
        return evaluator.evaluate(ctx.board, ctx.root_color)