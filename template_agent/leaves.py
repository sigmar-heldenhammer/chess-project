# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:46:49 2026

@author: Judson
"""

from .agent_templates import Evaluator, SearchContext
from typing import Optional


class DepthZeroLeaf:
    """Stop expanding only when depth==0; return static evaluator score."""
    def leaf_value(self, ctx: SearchContext, evaluator: Evaluator) -> Optional[float]:
        if ctx.depth > 0:
            return None
        return evaluator.evaluate(ctx.board, ctx.root_color)