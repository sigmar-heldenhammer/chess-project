# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:46:16 2025

@author: Judson
"""

# arena.py
import time
import io
import chess
import chess.pgn
from typing import Optional, Callable
from agents import Agent

def play_game(
    white: Agent,
    black: Agent,
    *,
    white_name: str = "White",
    black_name: str = "Black",
    time_control: Optional[tuple[float, float]] = None,  # (base, inc)
    pgn_out: Optional[io.TextIOBase] = None,
    quiet: bool = False,
    on_update: Optional[Callable[[chess.Board, chess.Move, int], None]] = None,  # <—
    on_root: Callable[[chess.Board, "Agent", "Agent"], None] | None = None
) -> dict:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["TimeControl"] = (
        f"{int(time_control[0])}+{int(time_control[1])}" if time_control else "-"
    )

    node = game

    if time_control:
        base, inc = time_control
        time_left = {chess.WHITE: base, chess.BLACK: base}
    else:
        inc = 0.0
        time_left = {chess.WHITE: None, chess.BLACK: None}

    agents = {chess.WHITE: white, chess.BLACK: black}
    ply = 0  # half-move index

    while not board.is_game_over(claim_draw=True):
        side = board.turn
        agent = agents[side]
        start = time.time()
        
        if on_root is not None:
            on_root(board.copy(stack=False), white, black)
            
        move = agent.select_move(
            board.copy(stack=True),
            time_left=time_left[side],
            orig_time = base,
            increment=inc,
            move_number=board.fullmove_number,
            color=side,
        )

        if move not in board.legal_moves:
            result = "0-1" if side == chess.WHITE else "1-0"
            termination = f"illegal move by {'White' if side else 'Black'}: {move.uci()}"
            game.headers["Termination"] = termination
            if not quiet:
                print("Illegal move; game forfeited:", termination)
            break

        board.push(move)
        ply += 1

        # 🔽 write an SVG (if a callback is provided)
        if on_update:
            on_update(board, move, ply)

        if time_control:
            elapsed = time.time() - start
            time_left[side] -= elapsed
            if time_left[side] < 0:
                result = "0-1" if side == chess.WHITE else "1-0"
                termination = f"time forfeit by {'White' if side else 'Black'}"
                game.headers["Termination"] = termination
                if not quiet:
                    print("Flag fell; game forfeited:", termination)
                break
            time_left[side] += inc

        node = node.add_variation(move)

    else:
        result = board.result(claim_draw=True)
        game.headers["Termination"] = describe_termination(board)

    game.headers["Result"] = result
    pgn_str = str(game)
    if pgn_out is not None:
        pgn_out.write(pgn_str + "\n\n")

    if not quiet:
        print("Final position:")
        print(board.unicode(borders=True))
        print("Result:", result, "-", game.headers.get("Termination", ""))

    return {
        "result": result,
        "termination": game.headers.get("Termination", ""),
        "moves": board.move_stack,
        "pgn": pgn_str,
    }

# ... describe_termination unchanged ...


def describe_termination(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "75-move rule"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.can_claim_threefold_repetition():
        return "threefold repetition (claimable)"
    if board.can_claim_fifty_moves():
        return "50-move rule (claimable)"
    if board.is_variant_draw():
        return "draw"
    return "game over"
