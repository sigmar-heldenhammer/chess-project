
"""
criteria_tester.py

Utilities to inspect and probe individual evaluation criteria methods
implemented inside agents (e.g., EvaluationAgent._crit_material_share).

New: can read the current board state from a FEN file (e.g., companion 'board.fen').
"""
from __future__ import annotations

import inspect
import chess
from typing import Any, Callable, Dict, List, Optional

# --- Import agents ---
try:
    from evaluation_agent import EvaluationAgent
except Exception:
    EvaluationAgent = None  # type: ignore

try:
    from minimax_agent import MinimaxAgent
except Exception:
    MinimaxAgent = None  # type: ignore

try:
    from greedy_material_agent import GreedyMaterialAgent
except Exception:
    GreedyMaterialAgent = None  # type: ignore

try:
    from random_agent import RandomAgent
except Exception:
    RandomAgent = None  # type: ignore

# Registry of agent names to classes
AGENT_REGISTRY: Dict[str, Any] = {
    "EvaluationAgent": EvaluationAgent,
    "MinimaxAgent": MinimaxAgent,
    "GreedyMaterialAgent": GreedyMaterialAgent,
    "RandomAgent": RandomAgent,
}

# Map friendly criterion aliases to private method names (extend as needed)
CRIT_ALIASES = {
    "material_share": "_crit_material_share",
    "center_control": "_crit_center_control",
}


def _resolve_agent(agent_name: str):
    cls = AGENT_REGISTRY.get(agent_name)
    if cls is None:
        raise ValueError(f"Unknown agent '{agent_name}'. Known: {', '.join(k for k,v in AGENT_REGISTRY.items() if v)}")
    if cls is None:
        raise ValueError(f"Agent '{agent_name}' is not available (import failed).")
    return cls


def list_criteria(agent_name: str) -> List[str]:
    """Return a list of '_crit_*' methods for the given agent."""
    cls = _resolve_agent(agent_name)
    obj = cls()  # type: ignore[call-arg]
    crits = []
    for name, fn in inspect.getmembers(obj, predicate=inspect.ismethod):
        if name.startswith("_crit_"):
            crits.append(name)
    return sorted(crits)


def _resolve_criterion(obj: Any, criterion: str) -> Callable[[chess.Board, chess.Color], float]:
    """Accept full private name or alias; return a bound method taking (board, color)."""
    if hasattr(obj, criterion):
        fn = getattr(obj, criterion)
        if callable(fn):
            return fn
    alias = CRIT_ALIASES.get(criterion, None)
    if alias and hasattr(obj, alias):
        fn = getattr(obj, alias)
        if callable(fn):
            return fn
    prefixed = f"_crit_{criterion}"
    if hasattr(obj, prefixed):
        fn = getattr(obj, prefixed)
        if callable(fn):
            return fn
    available = [n for n in dir(obj) if n.startswith("_crit_")]
    raise ValueError(f"Unknown criterion '{criterion}'. Try one of: {available}")


def _parse_color(color: Optional[str]) -> chess.Color:
    if color is None:
        return chess.WHITE
    low = color.lower().strip()
    if low in ("w", "white"):
        return chess.WHITE
    if low in ("b", "black"):
        return chess.BLACK
    raise ValueError("color must be 'white' or 'black' (or omitted).")


def _board_from_inputs(*, fen: Optional[str], fenfile: Optional[str]) -> chess.Board:
    """Construct a board from a FEN string or a FEN file path. Defaults to startpos."""
    if fenfile:
        try:
            with open(fenfile, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                if not line:
                    raise ValueError(f"Empty FEN file: {fenfile}")
                return chess.Board(line)
        except FileNotFoundError:
            raise FileNotFoundError(f"FEN file not found: {fenfile}")
    if fen:
        return chess.Board(fen)
    return chess.Board()


def evaluate_criterion(
    agent_name: str,
    criterion: str,
    *,
    fen: Optional[str] = None,
    fenfile: Optional[str] = None,
    board: Optional[chess.Board] = None,
    color: Optional[str] = "white",
    agent_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """Instantiate agent, resolve criterion, and evaluate the given board."""
    if board is None:
        b = _board_from_inputs(fen=fen, fenfile=fenfile)
    else:
        b = board
    cls = _resolve_agent(agent_name)
    kwargs = agent_kwargs or {}
    agent = cls(**kwargs)  # type: ignore[call-arg]
    fn = _resolve_criterion(agent, criterion)
    col = _parse_color(color)
    val = fn(b, col)
    return float(val)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Probe a single evaluation criterion on a given board.")
    parser.add_argument("--agent", required=True, help="Agent class name (e.g., EvaluationAgent)")
    parser.add_argument("--criterion", help="Criterion name or alias (e.g., center_control). Use --list to list all.")
    parser.add_argument("--fen", help="FEN string (omit for initial position)", default=None)
    parser.add_argument("--fenfile", help="Path to a file containing a single-line FEN (e.g., board.fen)", default=None)
    parser.add_argument("--from-live", action="store_true", help="Shorthand for --fenfile board.fen in current directory.")
    parser.add_argument("--color", help="white|black perspective", default="white")
    parser.add_argument("--list", action="store_true", help="List available criteria for this agent and exit.")
    parser.add_argument("--weights", help="Comma-list of weights for EvaluationAgent (e.g., 1,0.5)", default=None)
    parser.add_argument("--depth", type=int, default=None, help="Optional agent depth (if relevant)")

    args = parser.parse_args()

    if args.from_live and not args.fenfile:
        args.fenfile = "board.fen"

    if args.list:
        crits = list_criteria(args.agent)
        print("Available criteria:")
        for c in crits:
            print("  ", c)
    else:
        agent_kwargs = {}
        if args.depth is not None:
            agent_kwargs["depth"] = args.depth
        if args.weights is not None:
            try:
                weights = [float(x.strip()) for x in args.weights.split(",")]
                agent_kwargs["weights"] = weights
            except Exception:
                print("[tester] Could not parse --weights; expected comma separated floats.")
        try:
            v = evaluate_criterion(
                agent_name=args.agent,
                criterion=args.criterion,
                fen=args.fen,
                fenfile=args.fenfile,
                color=args.color,
                agent_kwargs=agent_kwargs,
            )
            print(f"Value: {v}")
        except Exception as e:
            print(f"[tester] Error: {e}")
