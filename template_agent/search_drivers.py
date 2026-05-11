# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 14:01:56 2026

@author: Judson
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .agent_templates import SearchContext, SearchResult, SearchDriver


@dataclass(frozen=True)
class SingleDepthDriver(SearchDriver):
    """Default driver: run exactly one root search at max_depth."""

    def run(
        self,
        *,
        max_depth: int,
        make_ctx: Callable[[int], SearchContext],
        search: Callable[[SearchContext], SearchResult],
    ) -> SearchResult:
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        return search(make_ctx(max_depth))


@dataclass(frozen=True)
class IterativeDeepeningDriver(SearchDriver):
    """Iterative deepening root driver.

    Runs depth=min_depth..max_depth and returns the result from the deepest
    completed iteration.

    TT/history should persist across iterations to improve move ordering.
    """

    min_depth: int = 1

    def run(
        self,
        *,
        max_depth: int,
        make_ctx: Callable[[int], SearchContext],
        search: Callable[[SearchContext], SearchResult],
    ) -> SearchResult:
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if self.min_depth < 0:
            raise ValueError(f"min_depth must be >= 0, got {self.min_depth}")

        if max_depth == 0:
            return search(make_ctx(0))

        start = min(self.min_depth, max_depth)

        best: Optional[SearchResult] = None
        for d in range(start, max_depth + 1):
            # print(d)
            res = search(make_ctx(d))
            best = res

            # No PV at root usually means terminal/no-legal-move scenario.
            if not res.pv:
                break

        assert best is not None
        return best
