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

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple, List, Dict, Any, Literal

import math
import random

import chess

try:
    # In your repo you likely have this.
    from agents import Agent  # type: ignore
except Exception:  # pragma: no cover
    class Agent:  # fallback so the file is importable standalone
        pass


# ----------------------------
# Shared helpers / constants
# ----------------------------

PIECE_VALUES: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
MATE_VALUE = 1000.0


def material_balance(board: chess.Board, root_color: chess.Color) -> float:
    """Material from root_color perspective, in pawns."""
    score = 0.0
    for piece_type, val in PIECE_VALUES.items():
        score += val * (
            len(board.pieces(piece_type, root_color))
            - len(board.pieces(piece_type, not root_color))
        )
    return score


# ----------------------------
# Search context and results
# ----------------------------

CutoffKind = Literal["none", "alpha", "beta"]
TTFlag = Literal["EXACT", "LOWER", "UPPER"]


@dataclass(frozen=True)
class SearchContext:
    board: chess.Board
    root_color: chess.Color
    maximizing: bool
    depth: int
    alpha: float
    beta: float
    # optional timing hooks (iterative deepening / depth adjustment later)
    time_left: Optional[float] = None
    orig_time: Optional[float] = None
    ply_from_root: int = 0


@dataclass(frozen=True)
class SearchResult:
    value: float
    pv: List[chess.Move]
    cutoff: CutoffKind
    best_move: Optional[chess.Move]


@dataclass(frozen=True)
class TTProbe:
    """Result of probing TT."""
    hit: bool
    value: Optional[float] = None
    flag: Optional[TTFlag] = None
    best_move_hint: Optional[chess.Move] = None
    stored_depth: int = 0


# ----------------------------
# Strategy interfaces (Protocols)
# ----------------------------

class Evaluator(Protocol):
    def evaluate(self, board: chess.Board, root_color: chess.Color) -> float: ...


class TerminalPolicy(Protocol):
    def terminal_value(self, board: chess.Board, root_color: chess.Color, evaluator: Evaluator) -> Optional[float]:
        """
        Return a numeric score if terminal (mate/draw/claim), else None.
        Keep terminal logic *history-aware* here (claim_draw=True).
        """


class LeafPolicy(Protocol):
    def leaf_value(
        self,
        ctx: SearchContext,
        evaluator: Evaluator,
    ) -> Optional[float]:
        """
        Return a numeric score if the node should stop expanding due to depth==0
        (or quiescence, etc.), else None.
        """


class OrderingPolicy(Protocol):
    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        ...

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: CutoffKind,
        depth: int,
    ) -> None:
        """Called when an alpha/beta cutoff occurs (history heuristic hook)."""


class TranspositionTable(Protocol):
    def probe(self, board: chess.Board, *, depth: int, alpha: float, beta: float) -> TTProbe: ...
    def store(
        self,
        board: chess.Board,
        *,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None: ...


class DepthPolicy(Protocol):
    def effective_depth(
        self,
        *,
        depth: int,
        orig_time: Optional[float],
        time_left: Optional[float],
    ) -> int:
        ...


# ----------------------------
# Default strategy implementations
# ----------------------------

class MaterialEvaluator:
    """Basic evaluator (non-terminal)."""
    def __init__(self, *, draw_contempt: float = 0.0, near_rep_nudge: float = 0.0):
        self.draw_contempt = float(draw_contempt)
        self.near_rep_nudge = float(near_rep_nudge)

    def evaluate(self, board: chess.Board, root_color: chess.Color) -> float:
        score = material_balance(board, root_color)

        # Optional "near repetition" soft nudge (kept mild by default).
        if self.near_rep_nudge != 0.0:
            try:
                if board.is_repetition(2) or board.can_claim_threefold_repetition():
                    if score >= 0.0:
                        score -= abs(self.near_rep_nudge)
                    else:
                        score += abs(self.near_rep_nudge)
            except Exception:
                pass

        return score


class DefaultTerminal:
    """
    Terminal check using python-chess' claim_draw=True.
    Uses evaluator material to apply optional draw contempt logic.
    """
    def __init__(self, *, mate_value: float = MATE_VALUE, draw_contempt: float = 0.0):
        self.mate_value = float(mate_value)
        self.draw_contempt = float(draw_contempt)

    def terminal_value(self, board: chess.Board, root_color: chess.Color, evaluator: Evaluator) -> Optional[float]:
        if not board.is_game_over(claim_draw=True):
            return None

        res = board.result(claim_draw=True)
        if res == "1-0":
            return self.mate_value if root_color == chess.WHITE else -self.mate_value
        if res == "0-1":
            return self.mate_value if root_color == chess.BLACK else -self.mate_value

        # Draw: optionally bias using contempt.
        if self.draw_contempt == 0.0:
            return 0.0

        mat = material_balance(board, root_color)
        # If we're better, dislike draws (negative). If worse, like draws (positive).
        return (-abs(self.draw_contempt)) if mat >= 0.0 else (+abs(self.draw_contempt))


class DepthZeroLeaf:
    """Stop expanding only when depth==0; return static evaluator score."""
    def leaf_value(self, ctx: SearchContext, evaluator: Evaluator) -> Optional[float]:
        if ctx.depth > 0:
            return None
        return evaluator.evaluate(ctx.board, ctx.root_color)


class NoTT:
    def probe(self, board: chess.Board, *, depth: int, alpha: float, beta: float) -> TTProbe:
        return TTProbe(hit=False)

    def store(
        self,
        board: chess.Board,
        *,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None:
        return


class SimpleDictTT:
    """
    A minimal TT that can be swapped in later.
    Stores (depth, value, flag, best_move) keyed by polyglot Zobrist hash.

    NOTE: This does not solve repetition/history draw issues; terminal checks must occur before probe.
    """
    def __init__(self):
        self._table: Dict[int, Tuple[int, float, TTFlag, Optional[chess.Move]]] = {}

    def _key(self, board: chess.Board) -> int:
        # polyglot hash is in python-chess
        return chess.polyglot.zobrist_hash(board)

    def probe(self, board: chess.Board, *, depth: int, alpha: float, beta: float) -> TTProbe:
        k = self._key(board)
        ent = self._table.get(k)
        if ent is None:
            return TTProbe(hit=False)

        stored_depth, value, flag, best_move = ent
        if stored_depth < depth:
            return TTProbe(hit=False, best_move_hint=best_move, stored_depth=stored_depth)

        # For now, we allow an EXACT hit to return a value.
        # Bounds can be used to tighten alpha/beta (engine-side) if you enable that later.
        return TTProbe(
            hit=True,
            value=value,
            flag=flag,
            best_move_hint=best_move,
            stored_depth=stored_depth,
        )

    def store(
        self,
        board: chess.Board,
        *,
        depth: int,
        value: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None:
        k = self._key(board)
        prev = self._table.get(k)
        if prev is None or depth >= prev[0]:
            self._table[k] = (depth, value, flag, best_move)


class BasicOrdering:
    """No history; optionally puts TT hint first."""
    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        out = list(moves)
        if tt_move_hint is not None and tt_move_hint in out:
            out.remove(tt_move_hint)
            out.insert(0, tt_move_hint)
        return out

    def on_cutoff(
        self,
        *,
        board: chess.Board,
        root_color: chess.Color,
        move: chess.Move,
        cutoff: CutoffKind,
        depth: int,
    ) -> None:
        return


class ActivityOrdering(BasicOrdering):
    """
    Mirrors your current “activity” heuristic style move ordering:
    captures/promotions/checks first.

    (Still a no-history baseline; history can be layered later by overriding on_cutoff + scoring.)
    """
    def _activity_score(self, board: chess.Board, mv: chess.Move) -> float:
        score = 0.0

        # If in check, prioritize evasions (boost all).
        if board.is_check():
            score += 1.0

        # Captures
        if board.is_capture(mv):
            # victim value
            victim_type = None
            if board.is_en_passant(mv):
                victim_type = chess.PAWN
            else:
                victim = board.piece_at(mv.to_square)
                victim_type = victim.piece_type if victim else None

            attacker = board.piece_at(mv.from_square)
            attacker_type = attacker.piece_type if attacker else None

            victim_val = PIECE_VALUES.get(victim_type, 0.0) if victim_type else 0.0
            attacker_val = PIECE_VALUES.get(attacker_type, 0.0) if attacker_type else 0.0

            score += 10.0 * victim_val - 0.1 * attacker_val

        # Promotions
        if mv.promotion is not None:
            score += PIECE_VALUES.get(mv.promotion, 0.0)

        # Giving check (requires push/pop)
        board.push(mv)
        try:
            if board.is_check():
                score += 0.5
        finally:
            board.pop()

        return score

    def order_moves(
        self,
        board: chess.Board,
        moves: Sequence[chess.Move],
        *,
        tt_move_hint: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        out = list(moves)
        scored = [(self._activity_score(board, mv), mv) for mv in out]
        scored.sort(key=lambda t: t[0], reverse=True)
        ordered = [mv for _, mv in scored]

        # Optional TT hint: force to front (useful if TT stores best_move).
        if tt_move_hint is not None and tt_move_hint in ordered:
            ordered.remove(tt_move_hint)
            ordered.insert(0, tt_move_hint)
        return ordered


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
        randomize_ties=False,
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
        randomize_ties=False,
        name=f"TTMinimax(d={depth})",
    )
