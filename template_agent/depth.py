# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:49:23 2026

@author: Judson
"""


from typing import Optional
import math

class NoDepthAdjustment:
    def effective_depth(self, *, depth: int, orig_time: Optional[float], time_left: Optional[float]) -> int:
        return max(int(depth), 0)


class LogDepthAdjustment:
    """
    Skeleton for your existing time_adjustment behavior.
    Kept separate so it doesn’t leak into the search loop logic.
    """
    def effective_depth(self, *, depth: int, orig_time: Optional[float], time_left: Optional[float]) -> int:
        d = int(depth)
        if orig_time is None or time_left is None:
            return max(d, 0)
        if orig_time <= 0 or time_left <= 0:
            return max(d, 0)
        if time_left >= orig_time:
            return max(d, 0)

        ratio = time_left / orig_time
        # Your existing logic: depth + ceil(log2(ratio))
        new_d = d + int(math.ceil(math.log2(ratio)))
        return max(new_d, 0)