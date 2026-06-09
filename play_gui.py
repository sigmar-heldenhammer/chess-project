# play_gui.py
"""
Executable local GUI entry point for playing against one of your existing agents.

This file wires together the GUI modules created so far:

    chess_gui_controller.py
    local_input.py
    human_gui_agent.py
    view_model.py
    board_renderer.py
    game_session.py

Purpose:
    Replace the previous text/SVG entry point with a minimal local graphical
    interface where the human selects moves by:

        left-click source square
        left-click destination square

Current intended use:
    Run this file directly:

        python play_gui.py

Expected architecture:

    pygame window / mouse
        ↓
    LocalMouseInputAdapter
        ↓
    ChessGUIController
        ↓
    HumanGUIAgent
        ↓
    arena.play_game(...)
        ↓
    GameSession.on_position_updated(...)
        ↓
    ViewModelBuilder
        ↓
    BoardRenderer

Important assumptions:
    1. Your existing project has:
           arena.play_game
           template_agent.make_qsearch_agent or another agent factory

    2. The generated GUI modules are importable from the same directory as this
       file, or are otherwise on PYTHONPATH.

    3. pygame and python-chess are installed in the active Python environment.

    4. arena.play_game runs synchronously and calls each Agent.select_move(...)
       when it needs a move.

    5. During a human turn, HumanGUIAgent.select_move(...) repeatedly calls
       app.pump_once(), which keeps the local pygame window responsive.

    6. During an engine turn, if the engine search is long-running and purely
       synchronous, the pygame window may temporarily stop updating. That is
       acceptable for this first iteration. Later, engine search can be moved
       into a worker thread/process or made incremental.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from board_renderer import BoardRenderer, create_pygame_board_window
from chess_gui_controller import ChessGUIController
from game_session import GameSession
from human_gui_agent import HumanGUIAgent, HumanGUIQuitRequested
from local_input import BoardGeometry, LocalMouseInputAdapter
from view_model import ViewModelBuilder


@dataclass
class ChessGUIAppConfig:
    """
    Configuration for the first local GUI.

    square_size:
        Pixel size of each square.

    board_left / board_top:
        Pixel offset of the board inside the window.

    white_at_bottom:
        Initial orientation.

    show_coordinates:
        Whether to draw file/rank coordinates.
        Defaults to False for the minimal first version.

    fps:
        Maximum pump/redraw frequency during human input.
    """

    square_size: int = 80
    board_left: int = 20
    board_top: int = 20
    white_at_bottom: bool = True
    show_coordinates: bool = False
    fps: int = 60


class ChessGUIApp:
    """
    Local pygame application shell.

    This class is the missing glue layer between the platform-independent
    controller and the platform-specific local renderer/input adapter.

    It deliberately does not contain chess engine logic. It only:
        - owns pygame window objects
        - polls local input
        - builds view models
        - redraws the board
        - exposes pump_once() for HumanGUIAgent
    """

    def __init__(
        self,
        *,
        white_agent_factory=None,
        black_agent_factory=None,
        config: Optional[ChessGUIAppConfig] = None,
        pgn_path: Optional[str] = "human_gui_game.pgn",
    ):
        """
        Inputs:
            white_agent_factory:
                Optional factory for the white player.

                If None, white is HumanGUIAgent.

                If supplied, it should be a zero-argument callable returning an
                Agent-like object.

            black_agent_factory:
                Optional factory for the black player.

                If None, defaults to make_qsearch_agent(depth=3, min_depth=-3).

            config:
                Optional ChessGUIAppConfig.

            pgn_path:
                File path for PGN output. Pass None to disable PGN output.

        Outputs:
            ChessGUIApp instance.

        Important:
            The default setup is human as White vs engine as Black.
        """
        self.config = config or ChessGUIAppConfig()

        self.controller = ChessGUIController()
        self.view_model_builder = ViewModelBuilder()

        # pygame/window/renderer/input are initialized together so they share
        # one BoardGeometry instance.
        self.surface, self.geometry, self.renderer = create_pygame_board_window(
            board_left=self.config.board_left,
            board_top=self.config.board_top,
            square_size=self.config.square_size,
            white_at_bottom=self.config.white_at_bottom,
            show_coordinates=self.config.show_coordinates,
        )

        self.input_adapter = LocalMouseInputAdapter(
            controller=self.controller,
            geometry=self.geometry,
        )

        self._pygame = self._load_pygame()
        self.clock = self._pygame.time.Clock()

        self.quit_requested = False

        # These are kept so pump_once() can render the most current board even
        # while HumanGUIAgent is blocking inside select_move(...).
        self.current_board = chess.Board()
        self.last_move: Optional[chess.Move] = None
        self.ply: int = 0

        self.pgn_path = pgn_path
        self.pgn_out = None

        # HumanGUIAgent needs pump_once() to keep the app responsive while it
        # waits for click input.
        self.human_agent = HumanGUIAgent(
            controller=self.controller,
            gui_pump=self.pump_once,
            should_quit=self.should_quit,
        )

        self.white = (
            white_agent_factory()
            if white_agent_factory is not None
            else self.human_agent
        )

        self.black = (
            black_agent_factory()
            if black_agent_factory is not None
            else self._default_black_agent()
        )

        self.session = GameSession(
            white=self.white,
            black=self.black,
            controller=self.controller,
            renderer=self.renderer,
            view_model_builder=self.view_model_builder,
            pgn_out=None,  # opened in run() so the handle can be closed safely
            initial_board=self.current_board,
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self):
        """
        Start the GUI game.

        Inputs:
            None

        Outputs:
            Whatever GameSession.start()/arena.play_game returns, if anything.

        Side effects:
            Opens pygame window and optional PGN output file.
        """
        try:
            if self.pgn_path is not None:
                self.pgn_out = open(self.pgn_path, "w", encoding="utf-8")
                self.session.pgn_out = self.pgn_out

            # Initial draw before arena.play_game begins.
            self.render_current_position()

            return self.session.start()

        except HumanGUIQuitRequested:
            # Normal user-closed-window path during a human move.
            return None

        finally:
            if self.pgn_out is not None:
                self.pgn_out.close()

            self._pygame.quit()

    # ------------------------------------------------------------------
    # GUI pump
    # ------------------------------------------------------------------

    def pump_once(self) -> None:
        """
        Process one GUI frame.

        Inputs:
            None

        Outputs:
            None

        Called by:
            HumanGUIAgent.select_move(...)

        Responsibilities:
            - poll local pygame events
            - feed square clicks into ChessGUIController
            - update current board reference from controller/session if needed
            - rebuild BoardViewModel
            - redraw BoardRenderer
            - throttle to configured FPS

        This method is intentionally small and explicit because it is the key
        integration point between blocking Agent.select_move(...) and the GUI.
        """
        input_result = self.input_adapter.handle_events()

        if input_result.quit_requested:
            self.quit_requested = True
            return

        self._sync_current_board()
        self.render_current_position()

        self.clock.tick(self.config.fps)

    def render_current_position(self) -> None:
        """
        Build and draw the current BoardViewModel.

        Inputs:
            None

        Outputs:
            None
        """
        view_model = self.view_model_builder.build_from_controller(
            board=self.current_board,
            controller=self.controller,
            last_move=self.last_move,
        )
        self.renderer.draw(view_model)

    def should_quit(self) -> bool:
        """
        Return whether the GUI has been asked to close.

        Inputs:
            None

        Outputs:
            bool
        """
        return self.quit_requested

    # ------------------------------------------------------------------
    # GameSession integration
    # ------------------------------------------------------------------

    def on_position_updated(
        self,
        board: chess.Board,
        move: Optional[chess.Move],
        ply: int,
    ) -> None:
        """
        Optional callback shape matching arena.play_game on_update.

        This method is provided in case you later want to bypass GameSession
        and pass app.on_position_updated directly to arena.play_game.

        In the current implementation, GameSession.on_position_updated is the
        callback used by play_game. GameSession updates its own state/rendering,
        while this app also keeps a local copy via _sync_current_board().
        """
        self.current_board = board
        self.last_move = move
        self.ply = ply
        self.controller.set_board(board)
        self.render_current_position()

    def _sync_current_board(self) -> None:
        """
        Keep app.current_board aligned with the best available source.

        During human turns:
            HumanGUIAgent.select_move(board) calls controller.set_board(board),
            so controller.board is the most up-to-date board.

        After completed moves:
            GameSession stores the latest board.

        This method avoids stale rendering while the human agent is blocking.
        """
        if self.controller.board is not None:
            self.current_board = self.controller.board

        elif self.session.board is not None:
            self.current_board = self.session.board

        self.last_move = self.session.last_move
        self.ply = self.session.ply

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _default_black_agent(self):
        """
        Create the default engine opponent.

        Outputs:
            Agent-like object.

        Adjust this to use whichever of your existing agent factories you want.
        """
        try:
            from template_agent import make_qsearch_agent

            return make_qsearch_agent(depth=1, min_depth=-1)
        except ImportError:
            # Fallback to a simpler factory name if the project setup differs.
            from template_agent import make_basic_minimax

            return make_basic_minimax(depth=3)

    # ------------------------------------------------------------------
    # pygame loading
    # ------------------------------------------------------------------

    def _load_pygame(self):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "play_gui.py requires pygame. Install pygame in your active "
                "environment before running the local GUI."
            ) from exc

        return pygame


def main() -> None:
    """
    Default executable entry point.

    Human plays White against the default qsearch agent as Black.
    """
    app = ChessGUIApp()
    app.run()


if __name__ == "__main__":
    main()
