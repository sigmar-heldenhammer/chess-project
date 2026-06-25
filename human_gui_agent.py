# human_gui_agent.py
"""
Human GUI agent for the chess GUI.

This module adapts the click-based ChessGUIController to the existing Agent
interface used by the current game loop.

Current CLI pattern:
    HumanCLI.select_move(board, **kwargs) blocks until the user enters a move.

This GUI pattern:
    HumanGUIAgent.select_move(board, **kwargs) blocks until the user completes
    a legal move by clicking source square -> destination square.

Important design choice:
    This file does not import pygame and does not know about pixels, windows,
    or rendering. It depends on an injected "gui pump" callback to keep the
    local frontend responsive while waiting for a move.

Expected architecture while waiting for human input:

    HumanGUIAgent.select_move(board)
        ↓
    controller.set_board(board)
        ↓
    repeatedly call gui_pump()
        ↓
    LocalMouseInputAdapter.handle_events()
        ↓
    ChessGUIController.handle_square_click(square)
        ↓
    controller.pop_pending_move()
        ↓
    return chess.Move to play_game(...)

Assumptions about not-yet-implemented functionality:
    1. Your existing agents inherit from agents.Agent and expose:
           select_move(self, board: chess.Board, **kwargs) -> chess.Move

    2. ChessGUIController exists in chess_gui_controller.py and exposes:
           set_board(board)
           pop_pending_move()
           clear_selection()

    3. A local app object or function will be passed as gui_pump. That function
       should process local input events and redraw the board.

       For example, a future ChessGUIApp might expose:
           app.pump_once()

       where pump_once does:
           input_result = input_adapter.handle_events()
           if input_result.quit_requested:
               app.request_quit()
           view_model = view_model_builder.build(...)
           renderer.draw(view_model)

    4. If the GUI window is closed while waiting for human input, gui_pump may
       raise KeyboardInterrupt/SystemExit, or the optional should_quit callback
       may return True. In that case this agent raises HumanGUIQuitRequested.

    5. The external game loop remains responsible for board.push(move). This
       agent only returns the selected move.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import chess

try:
    from agents import Agent
except ImportError:
    # Fallback so this file can be imported/tested before being placed inside
    # the full project. In the actual project, agents.Agent should be available.
    class Agent:  # type: ignore[no-redef]
        def select_move(self, board: chess.Board, **kwargs) -> chess.Move:
            raise NotImplementedError


class HumanGUIQuitRequested(RuntimeError):
    """
    Raised when the user closes the GUI while HumanGUIAgent is waiting for input.
    """


class ChessGUIControllerProtocol(Protocol):
    """
    Minimal controller protocol required by HumanGUIAgent.

    Using a Protocol avoids tightly coupling this file to one concrete
    controller implementation.
    """

    def set_board(self, board: chess.Board) -> None:
        ...

    def pop_pending_move(self) -> Optional[chess.Move]:
        ...

    def clear_selection(self, keep_message: bool = False) -> None:
        ...


GUIPump = Callable[[], None]
ShouldQuit = Callable[[], bool]
ShouldConcede = Callable[[], bool]


class HumanGUIAgent(Agent):
    """
    Human player adapter for GUI-based move input.

    The class preserves your existing engine/game-loop expectation that each
    player is an Agent with select_move(...).

    Instead of asking for SAN/UCI text, it waits for ChessGUIController to
    produce a pending move from square-click input.
    """

    def __init__(
        self,
        controller: ChessGUIControllerProtocol,
        gui_pump: GUIPump,
        *,
        should_quit: Optional[ShouldQuit] = None,
        should_concede: Optional[ShouldConcede] = None,
        name: str = "HumanGUI",
    ):
        """
        Inputs:
            controller:
                ChessGUIController or compatible object.

            gui_pump:
                Callback that keeps the GUI alive while waiting for input.
                It should process events and redraw once.

            should_quit:
                Optional callback returning True if the app/window is closing.

            should_concede:
                Optional callback returning True if the app requested that the
                human player concede the current game.

            name:
                Human-readable agent name.

        Outputs:
            HumanGUIAgent instance.
        """
        self.controller = controller
        self.gui_pump = gui_pump
        self.should_quit = should_quit
        self.should_concede = should_concede
        self.name = name

    def select_move(self, board: chess.Board, **kwargs) -> chess.Move:
        """
        Wait for a legal GUI-selected move and return it.

        Inputs:
            board:
                Current python-chess Board supplied by the existing play_game
                loop.

            **kwargs:
                Ignored for now, but accepted for compatibility with the
                existing Agent interface and future time-control metadata.

        Outputs:
            chess.Move:
                A legal move selected by the human.

        Raises:
            HumanGUIQuitRequested:
                If the GUI is closed while waiting.

        Notes:
            - The controller may already hold the same mutable board reference,
              but set_board(board) is called here to make synchronization
              explicit at the start of every human turn.
            - This method does not push the move. The existing game loop should
              do that after select_move returns.
        """
        self.controller.set_board(board)

        while True:
            if self.should_quit is not None and self.should_quit():
                self.controller.clear_selection()
                raise HumanGUIQuitRequested("GUI closed while waiting for human move.")

            if self.should_concede is not None and self.should_concede():
                self.controller.clear_selection()
                return chess.Move.null()

            # Keep frontend responsive. This is where the app should:
            #   - handle input events
            #   - let LocalMouseInputAdapter feed square clicks to controller
            #   - rebuild view model
            #   - redraw renderer
            self.gui_pump()

            if self.should_concede is not None and self.should_concede():
                self.controller.clear_selection()
                return chess.Move.null()

            move = self.controller.pop_pending_move()

            if move is None:
                continue

            if move in board.legal_moves:
                return move

            # Defensive: pending_move should already be legal because the
            # controller validates it, but the board may have changed or the
            # controller may be reused incorrectly.
            self.controller.clear_selection()

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
