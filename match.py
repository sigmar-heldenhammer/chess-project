# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:14:34 2025

@author: Judson
"""

# match.py
import io
import time
import chess
from typing import Optional, Callable
from arena import play_game
from agents import Agent

# --- Timing wrapper -----------------------------------------------------------
class TimedAgent:
    """
    Wraps an agent and records the cumulative time spent inside select_move,
    plus how many moves were made. Acts as a transparent proxy otherwise.
    """
    def __init__(self, inner):
        self.inner = inner
        self.total_time = 0.0
        self.move_count = 0

    def select_move(self, board, **kwargs):
        start = time.time()
        move = self.inner.select_move(board, **kwargs)
        self.total_time += (time.time() - start)
        self.move_count += 1
        return move

    # Forward any other attributes (name, config, etc.) to the inner agent
    def __getattr__(self, name):
        return getattr(self.inner, name)


def play_match(a, b, games: int = 20, tc=(60, 0), pgn_path: str | None = "match.pgn", 
    divergence_probe: Callable[[chess.Board, "Agent", "Agent"], None] | None = None):
    score = 0.0  # positive = a leads; negative = b leads
    wdl = [0, 0, 0]  # [wins for a, draws, wins for b]
    pgn_out = open(pgn_path, "w", encoding="utf-8") if pgn_path else None

    # Wrap once so timing spans all games
    ta = TimedAgent(a)
    tb = TimedAgent(b)

    try:
        for i in range(games):
            if i % 2 == 0:
                white, black = ta, tb
                wn, bn = "A", "B"
            else:
                white, black = tb, ta
                wn, bn = "B", "A"

            res = play_game(
                white=white, black=black,
                white_name=f"{wn}", black_name=f"{bn}",
                time_control=tc, pgn_out=pgn_out, quiet=True,
                on_root = divergence_probe
            )

            r = res["result"]
            if r == "1-0":
                score += 1.0 if wn == "A" else -1.0
                wdl[0 if wn == "A" else 2] += 1
            elif r == "0-1":
                score += -1.0 if wn == "A" else 1.0
                wdl[2 if wn == "A" else 0] += 1
            else:
                score += 0.0
                wdl[1] += 1

            print(f"Game {i+1}/{games}: {r} | Termination: {res['termination']}")

        print(f"\nFinal score A vs B: {score:.1f} (W-D-L A): {wdl[0]}-{wdl[1]}-{wdl[2]}")
        if pgn_out:
            print(f"PGN saved to: {pgn_out.name}")

        # --- Timing summary ---------------------------------------------------
        def avg_s(agent: TimedAgent) -> Optional[float]:
            return (agent.total_time / agent.move_count) if agent.move_count > 0 else None

        a_avg = avg_s(ta)
        b_avg = avg_s(tb)

        def fmt(avg):
            return f"{avg*1000:.1f} ms" if avg is not None else "n/a"

        print("\nAverage time per move (across all games):")
        print(f"  Agent A: {fmt(a_avg)}  (moves: {ta.move_count})")
        print(f"  Agent B: {fmt(b_avg)}  (moves: {tb.move_count})")

    finally:
        if pgn_out:
            pgn_out.close()


if __name__ == "__main__":
    A = RandomAgent(seed=1)
    B = RandomAgent(seed=2)
    play_match(A, B, games=10, tc=(10, 0))  # blitzy self-play
