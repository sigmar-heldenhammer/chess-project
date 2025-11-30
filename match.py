# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:14:34 2025

@author: Judson
"""

# match.py
import io
from arena import play_game
from random_agent import RandomAgent

def play_match(a, b, games: int = 20, tc=(60, 0), pgn_path: str | None = "match.pgn"):
    score = 0.0  # positive = a leads; negative = b leads
    wdl = [0, 0, 0]  # [wins for a, draws, wins for b]
    pgn_out = open(pgn_path, "w", encoding="utf-8") if pgn_path else None

    try:
        for i in range(games):
            if i % 2 == 0:
                white, black = a, b
                wn, bn = "A", "B"
            else:
                white, black = b, a
                wn, bn = "B", "A"

            res = play_game(
                white=white, black=black,
                white_name=f"{wn}", black_name=f"{bn}",
                time_control=tc, pgn_out=pgn_out, quiet=True
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
    finally:
        if pgn_out: pgn_out.close()

if __name__ == "__main__":
    A = RandomAgent(seed=1)
    B = RandomAgent(seed=2)
    play_match(A, B, games=10, tc=(10, 0))  # blitzy self-play
