
"""
Created on Tue Dec  9 01:40:21 2025

@author: Judson
"""

"""
divergence_logger.py

Compare two agents at the current root position and log a JSON record when their
chosen moves diverge (optionally requiring a minimum score gap). Designed to work
with agents that expose after select_move():
    - last_root_best_move: chess.Move
    - last_root_score: float
    - last_root_pv: list[chess.Move]
    - depth, use_alpha_beta, order_moves, seed (optional, for metadata)
"""

#from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import chess

_COUNTER = 0  # module-level counter per process


def _agent_snapshot(agent) -> Dict[str, Any]:
    def to_uci_list(pv):
        try:
            return [m.uci() for m in pv] if pv else []
        except Exception:
            return []

    return {
        "class": type(agent).__name__,
        "seed": getattr(agent, "seed", None),
        "depth": getattr(agent, "depth", None),
        "alpha_beta": getattr(agent, "use_alpha_beta", None),
        "order_moves": getattr(agent, "order_moves", None),
        "move": getattr(agent, "last_root_best_move", None).uci() if getattr(agent, "last_root_best_move", None) else None,
        "score": getattr(agent, "last_root_score", None),
        "pv": to_uci_list(getattr(agent, "last_root_pv", None)),
    }


def compare_agents_at_root(
    board: chess.Board,
    agent_a,
    agent_b,
    *,
    score_gap: float = 0.20,
    out_dir: str = "divergences",
    log_limit: int = 50,
    sample_rate: float = 1.0,
    include_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run both agents from the same root and log when their chosen moves diverge.

    - score_gap: minimum absolute difference in root scores to log (in "pawn" units).
    - out_dir: directory for JSON logs.
    - log_limit: max files to write per process import.
    - sample_rate: probability in [0,1] to perform the comparison (for overhead control).
    - include_meta: optional dict to merge into the JSON (e.g., match/game IDs).
    """
    import random
    global _COUNTER
    if _COUNTER >= log_limit:
        return
    if sample_rate < 1.0 and random.random() > sample_rate:
        return

    # Evaluate both agents on *copies* of the board to avoid state pollution.
    b1 = chess.Board(board.fen())
    b2 = chess.Board(board.fen())

    color = b1.turn

    # Ask each agent for its move; they will fill last_root_* fields.
    try:
        _ = agent_a.select_move(b1, color=color)
    except TypeError:
        _ = agent_a.select_move(b1)
    try:
        _ = agent_b.select_move(b2, color=color)
    except TypeError:
        _ = agent_b.select_move(b2)

    a_snap = _agent_snapshot(agent_a)
    b_snap = _agent_snapshot(agent_b)

    # If either failed to supply a move or score, skip.
    if not a_snap["move"] or not b_snap["move"]:
        return

    # Only log when they actually choose different moves
    if a_snap["move"] == b_snap["move"]:
        return

    # If both have scores, enforce a minimum gap to avoid noisy ties
    a_score = a_snap["score"]
    b_score = b_snap["score"]
    if a_score is not None and b_score is not None:
        if abs(a_score - b_score) < score_gap:
            return

    payload = {
        "root_fen": board.fen(),
        "side_to_move": "white" if color == chess.WHITE else "black",
        "settings": {
            "score_gap": score_gap,
            "sample_rate": sample_rate,
        },
        "agent_a": a_snap,
        "agent_b": b_snap,
        "score_diff": None if (a_score is None or b_score is None) else abs(a_score - b_score),
    }

    if include_meta:
        payload["meta"] = dict(include_meta)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _COUNTER += 1
    path = Path(out_dir) / f"div_{_COUNTER:03d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
