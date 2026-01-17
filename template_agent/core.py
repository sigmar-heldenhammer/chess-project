# modular_search_agent.py
# First-draft “Strategy + Template” minimax framework:
# - One canonical alpha-beta search loop (no duplication)
# - Pluggable strategies: Evaluation, Terminal, Leaf, Ordering(+history hooks), TT, Depth policy
# - Defaults implement a basic minimax agent equivalent in spirit to your current MinimaxAgent
#
# Notes:
# - This is a framework draft: intentionally conservative and readable.
# - It supports TT + history hooks structurally, but by default uses NoTT + basic ordering.
# - PV is always constructed by the search loop (TT returns only hints / cached scores, never PV).

from __future__ import annotations


from typing import Optional, List, Any
from .agent_templates import SearchContext, SearchResult, Evaluator, CutoffKind, TTFlag, \
    TerminalPolicy, LeafPolicy, OrderingPolicy, TranspositionTable, DepthPolicy
from .evals import MaterialEvaluator
from .terminals import DefaultTerminal
from .leaves import DepthZeroLeaf
from .ordering import ActivityOrdering
from .tt import NoTT
from .depth import NoDepthAdjustment

import random

import chess
try:
    import chess.polyglot
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "minimax_agent_tt.py requires python-chess with polyglot support."
    ) from e

try:
    # In your repo you likely have this.
    from agents import Agent  # type: ignore
except Exception:  # pragma: no cover
    class Agent:  # fallback so the file is importable standalone
        pass



# ----------------------------
# The modular search agent (single canonical loop)
# ----------------------------

class ModularMinimaxAgent(Agent):
    """
    A configurable alpha-beta minimax agent.

    Strategies:
      - evaluator: static evaluation (non-terminal)
      - terminal: mate/draw/claim logic
      - leaf: depth==0 handling (static now; quiescence later)
      - ordering: move ordering + optional on_cutoff hook (history later)
      - tt: transposition table probe/store (NoTT by default)
      - depth_policy: time-based effective depth (NoDepthAdjustment by default)
    """
    def __init__(
        self,
        *,
        depth: int = 3,
        evaluator: Optional[Evaluator] = None,
        terminal: Optional[TerminalPolicy] = None,
        leaf: Optional[LeafPolicy] = None,
        ordering: Optional[OrderingPolicy] = None,
        tt: Optional[TranspositionTable] = None,
        depth_policy: Optional[DepthPolicy] = None,
        randomize_ties: bool = False,
        seed: Optional[int] = None,
        name: str = "ModularMinimaxAgent",
        **kwargs: Any,
    ):
        super().__init__()  # if Agent base has init; harmless otherwise

        self.name = name
        self.depth = int(depth)

        self.evaluator: Evaluator = evaluator or MaterialEvaluator()
        self.terminal: TerminalPolicy = terminal or DefaultTerminal()
        self.leaf: LeafPolicy = leaf or DepthZeroLeaf()
        self.ordering: OrderingPolicy = ordering or ActivityOrdering()
        self.tt: TranspositionTable = tt or NoTT()
        self.depth_policy: DepthPolicy = depth_policy or NoDepthAdjustment()

        self.randomize_ties = bool(randomize_ties)
        if seed is not None:
            random.seed(seed)

        # Last-search diagnostics (kept compatible with your current style)
        self.last_root_best_move: Optional[chess.Move] = None
        self.last_root_score: Optional[float] = None
        self.last_root_pv: List[chess.Move] = []

    # ---- public API expected by arena ----
    def select_move(
        self,
        board: chess.Board,
        *,
        color: Optional[chess.Color] = None,
        time_left: Optional[float] = None,
        orig_time: Optional[float] = None,
        **kwargs: Any,
    ) -> chess.Move:
        root_color = board.turn if color is None else color

        eff_depth = self.depth_policy.effective_depth(
            depth=self.depth,
            orig_time=orig_time,
            time_left=time_left,
        )

        # Root search
        ctx = SearchContext(
            board=board,
            root_color=root_color,
            maximizing=True,
            depth=eff_depth,
            alpha=float("-inf"),
            beta=float("+inf"),
            time_left=time_left,
            orig_time=orig_time,
            ply_from_root=0,
        )
        res = self._search(ctx)

        # Choose root move
        if not res.pv:
            # Fallback: no PV (should only happen if no legal moves)
            moves = list(board.legal_moves)
            if not moves:
                raise ValueError("No legal moves available.")
            move = moves[0]
            self.last_root_best_move = move
            self.last_root_score = res.value
            self.last_root_pv = []
            return move

        best_move = res.pv[0]
        self.last_root_best_move = best_move
        self.last_root_score = res.value
        self.last_root_pv = list(res.pv)
        return best_move

    # ---- canonical alpha-beta loop (do not duplicate in subclasses) ----
    def _search(self, ctx: SearchContext) -> SearchResult:
        board = ctx.board

        # 1) Terminal check (history-aware). Do this before TT.
        term = self.terminal.terminal_value(board, ctx.root_color, self.evaluator)
        if term is not None:
            return SearchResult(value=term, pv=[], cutoff="none", best_move=None)

        # 2) Leaf check (depth==0 / quiescence hook). Do this before TT.
        leaf_val = self.leaf.leaf_value(ctx, self.evaluator)
        if leaf_val is not None:
            return SearchResult(value=leaf_val, pv=[], cutoff="none", best_move=None)

        # 3) TT probe (history-blind, but safe now that terminal/leaf already handled).
        tt_probe = self.tt.probe(board, depth=ctx.depth, alpha=ctx.alpha, beta=ctx.beta)
        tt_hint = tt_probe.best_move_hint

        # For now: only return directly on EXACT hits.
        if tt_probe.hit and tt_probe.value is not None and tt_probe.flag == "EXACT":
            return SearchResult(value=tt_probe.value, pv=[], cutoff="none", best_move=tt_hint)

        # 4) Generate and order moves
        moves = list(board.legal_moves)
        if not moves:
            # No legal moves; terminal_value should have caught mate/stalemate,
            # but keep safe fallback.
            val = self.evaluator.evaluate(board, ctx.root_color)
            return SearchResult(value=val, pv=[], cutoff="none", best_move=None)

        ordered = self.ordering.order_moves(board, moves, tt_move_hint=tt_hint)

        # 5) Recurse
        best_val = float("-inf") if ctx.maximizing else float("+inf")
        best_pvs: List[List[chess.Move]] = []
        best_move: Optional[chess.Move] = None

        alpha = ctx.alpha
        beta = ctx.beta

        cutoff_kind: CutoffKind = "none"
        stored_flag: TTFlag = "EXACT"

        for mv in ordered:
            board.push(mv)
            try:
                child_ctx = SearchContext(
                    board=board,
                    root_color=ctx.root_color,
                    maximizing=not ctx.maximizing,
                    depth=ctx.depth - 1,
                    alpha=alpha,
                    beta=beta,
                    time_left=ctx.time_left,
                    orig_time=ctx.orig_time,
                    ply_from_root=ctx.ply_from_root + 1,
                )
                child_res = self._search(child_ctx)
            finally:
                board.pop()

            child_val = child_res.value
            child_pv = child_res.pv

            pv_here = [mv] + child_pv

            if ctx.maximizing:
                if child_val > best_val:
                    best_val = child_val
                    best_move = mv
                    best_pvs = [pv_here]
                elif child_val == best_val:
                    best_pvs.append(pv_here)

                # alpha update
                if best_val > alpha:
                    alpha = best_val

                # beta cutoff
                if alpha >= beta:
                    cutoff_kind = "beta"
                    stored_flag = "LOWER"
                    self.ordering.on_cutoff(
                        board=ctx.board,
                        root_color=ctx.root_color,
                        move=mv,
                        cutoff=cutoff_kind,
                        depth=ctx.depth,
                    )
                    break

            else:
                if child_val < best_val:
                    best_val = child_val
                    best_move = mv
                    best_pvs = [pv_here]
                elif child_val == best_val:
                    best_pvs.append(pv_here)

                # beta update (minimizer tightens beta)
                if best_val < beta:
                    beta = best_val

                # alpha cutoff
                if alpha >= beta:
                    cutoff_kind = "alpha"
                    stored_flag = "UPPER"
                    self.ordering.on_cutoff(
                        board=ctx.board,
                        root_color=ctx.root_color,
                        move=mv,
                        cutoff=cutoff_kind,
                        depth=ctx.depth,
                    )
                    break

        # Choose PV among ties (optional randomization).
        if not best_pvs:
            # Shouldn’t happen (we had moves), but keep safe.
            best_pv = []
        else:
            best_pv = random.choice(best_pvs) if self.randomize_ties else best_pvs[0]

        # 6) TT store (at every node), using bound kind if cut off.
        # NOTE: Even if you start with exact-only, this signature is ready for bounds.
        self.tt.store(
            board=ctx.board,
            depth=ctx.depth,
            value=best_val,
            flag=stored_flag,
            best_move=best_move,
        )

        return SearchResult(value=best_val, pv=best_pv, cutoff=cutoff_kind, best_move=best_move)


