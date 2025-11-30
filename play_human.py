# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:48:39 2025

@author: Judson
"""

# play_human.py
import chess, chess.svg
from arena import play_game
from random_agent import RandomAgent
from greedy_material_agent  import GreedyMaterialAgent
from minimax_agent  import MinimaxAgent
from evaluation_agent import EvaluationAgent

from human_cli_agent import HumanCLI

def write_svg(board: chess.Board, move: chess.Move, ply: int):
    svg = chess.svg.board(board, coordinates=True, lastmove=move)

    with open("board.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    if ply == 1:
        print("SVG writing to board.svg — open it in your browser and refresh as you play.")
    with open("board.fen", "w", encoding="utf-8") as f:
        f.write(board.fen())


if __name__ == "__main__":
    play_game(
        white=HumanCLI(),

        black=EvaluationAgent(depth=2, seed=42, use_alpha_beta=False, weights=[1.0,1.0, 1.0]),
        time_control=None,
        on_update=write_svg,   # <— enable SVG output
    )

"""

"""