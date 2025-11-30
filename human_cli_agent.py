# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:48:03 2025

@author: Judson
"""

# human_cli_agent.py
import chess
from agents import Agent

class HumanCLI(Agent):
    def select_move(self, board: chess.Board, **kwargs) -> chess.Move:
        # show a simple board & prompt SAN or UCI
        print(board.unicode(borders=True))
        while True:
            s = input("Your move (SAN like 'Nf3' or UCI like 'g1f3'): ").strip()
            try:
                # try SAN first
                try:
                    mv = board.parse_san(s)
                except ValueError:
                    mv = chess.Move.from_uci(s)
                    if mv not in board.legal_moves:
                        raise ValueError
                return mv
            except Exception:
                print("Invalid move. Try again.")
