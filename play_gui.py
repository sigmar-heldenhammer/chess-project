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

from dataclasses import dataclass, field, replace
from typing import Optional
from enum import Enum

import chess

from board_renderer import BoardRenderer, create_pygame_board_window
from chess_gui_controller import ChessGUIController
from game_session import GameSession
from human_gui_agent import HumanGUIAgent, HumanGUIQuitRequested
from local_input import BoardGeometry, LocalMouseInputAdapter, PromotionMenuGeometry
from view_model import ViewModelBuilder

class BoardPerspective(Enum):
    WHITE_BOTTOM = "white_bottom"
    PLAYER_BOTTOM = "player_bottom"


class AppMode(Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"
    QUIT_REQUESTED = "quit_requested"


@dataclass
class WindowConfig:
    square_size: int = 80
    board_left: int = 20
    board_top: int = 20


@dataclass
class DisplayConfig:
    perspective: BoardPerspective = BoardPerspective.WHITE_BOTTOM
    show_coordinates: bool = True


@dataclass
class PlayerConfig:
    human_color: chess.Color = chess.WHITE
    white_display_name: Optional[str] = None
    black_display_name: Optional[str] = None


@dataclass
class ControlConfig:
    fps: int = 60


@dataclass
class ChessGUIConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    players: PlayerConfig = field(default_factory=PlayerConfig)
    controls: ControlConfig = field(default_factory=ControlConfig)




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
        config: Optional[ChessGUIConfig] = None,
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
                Optional ChessGUIConfig.

            pgn_path:
                File path for PGN output. Pass None to disable PGN output.

        Outputs:
            ChessGUIApp instance.

        Important:
            The default setup is human as White vs engine as Black.
        """
        self.config = config or ChessGUIConfig()
        self.white_agent_factory = white_agent_factory
        self.black_agent_factory = black_agent_factory

        self.view_model_builder = ViewModelBuilder()

        # pygame/window/renderer/input are initialized together so they share
        # one BoardGeometry instance.
        self.surface, self.geometry, self.renderer = create_pygame_board_window(
            board_left=self.config.window.board_left,
            board_top=self.config.window.board_top,
            square_size=self.config.window.square_size,
            white_at_bottom=True,
            show_coordinates=self.config.display.show_coordinates,
        )

        self.apply_board_perspective()

        self.promotion_menu_geometry = PromotionMenuGeometry(
            board_geometry=self.geometry,
            option_size=self.geometry.square_size,
        )

        self.renderer.promotion_menu_geometry = self.promotion_menu_geometry

        self._pygame = self._load_pygame()
        self.clock = self._pygame.time.Clock()

        self.quit_requested = False
        self.app_mode = AppMode.PLAYING
        self.post_game_overlay_visible = True
        self.concede_requested = False
        self.conceding_color: Optional[chess.Color] = None

        self.controller: ChessGUIController
        self.input_adapter: LocalMouseInputAdapter
        self.current_board: chess.Board
        self.last_move: Optional[chess.Move]
        self.ply: int
        self.pgn_path = pgn_path
        self.pgn_out = None
        self.human_agent = None
        self.white = None
        self.black = None
        self.white_display_name = "White"
        self.black_display_name = "Black"
        self.session: Optional[GameSession] = None

        self.start_new_game()

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
        last_result = None

        try:
            if self.pgn_path is not None:
                self.pgn_out = open(self.pgn_path, "w", encoding="utf-8")
                self.session.pgn_out = self.pgn_out

            while not self.quit_requested:
                assert self.session is not None
                self.session.pgn_out = self.pgn_out
                self.app_mode = AppMode.PLAYING

                # Initial draw before arena.play_game begins.
                self.render_current_position()

                last_result = self.session.start()
                self._apply_concession_post_game_if_needed()

                if self.quit_requested:
                    break

                self.app_mode = AppMode.GAME_OVER
                action = self.run_game_over_screen()

                if action == "rematch":
                    self.start_new_game()
                    continue

                break

            return last_result

        except HumanGUIQuitRequested:
            # Normal user-closed-window path during a human move.
            return None

        finally:
            self.app_mode = AppMode.QUIT_REQUESTED
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
        self._handle_input_result(input_result)

        if self.quit_requested:
            return

        self._sync_current_board()
        self.render_current_position()

        self.clock.tick(self.config.controls.fps)

    def run_game_over_screen(self) -> Optional[str]:
        while not self.quit_requested:
            input_result = self.input_adapter.handle_events()
            action = self._handle_input_result(input_result)

            if action in {"rematch", "quit"}:
                return action

            self._sync_current_board()
            self.render_current_position()
            self.clock.tick(self.config.controls.fps)

        return "quit"

    def _handle_input_result(self, input_result) -> Optional[str]:
        if input_result.window_resized and input_result.window_size is not None:
            width, height = input_result.window_size
            self.renderer.resize_to_window(width, height)
            self.promotion_menu_geometry.option_size = self.geometry.square_size

        if input_result.scroll_delta_y != 0:
            self.renderer.handle_move_tracker_scroll(
                input_result.scroll_delta_y,
                self._pygame.mouse.get_pos(),
            )

        if input_result.quit_requested:
            self.quit_requested = True
            self.app_mode = AppMode.QUIT_REQUESTED
            return "quit"

        if input_result.ui_action == "quit":
            self.quit_requested = True
            self.app_mode = AppMode.QUIT_REQUESTED
            return "quit"

        if input_result.ui_action == "rematch":
            return "rematch"

        if input_result.ui_action == "concede":
            self.request_concede()
            return "concede"

        if input_result.ui_action == "close_post_game":
            self.post_game_overlay_visible = False
            return "close_post_game"

        return input_result.ui_action

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
            white_display_name=self.white_display_name,
            black_display_name=self.black_display_name,
            post_game=self._current_post_game_view(),
        )
        self.renderer.draw(view_model)

    def get_latest_view_model(self):
        self._sync_current_board()
        return self.view_model_builder.build_from_controller(
            board=self.current_board,
            controller=self.controller,
            last_move=self.last_move,
            white_display_name=self.white_display_name,
            black_display_name=self.black_display_name,
            post_game=self._current_post_game_view(),
        )

    def _current_post_game_view(self):
        if self.session is None or self.session.post_game is None:
            return None

        return replace(
            self.session.post_game,
            show_overlay=self.post_game_overlay_visible,
        )

    def should_quit(self) -> bool:
        """
        Return whether the GUI has been asked to close.

        Inputs:
            None

        Outputs:
            bool
        """
        return self.quit_requested

    def should_concede(self) -> bool:
        return self.concede_requested

    def request_concede(self) -> None:
        if self.app_mode != AppMode.PLAYING:
            return

        self.concede_requested = True
        self.conceding_color = (
            self.current_board.turn
            if self.current_board is not None
            else self.config.players.human_color
        )

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
        if (
            self.app_mode == AppMode.GAME_OVER
            and self.session is not None
            and self.session.board is not None
        ):
            self.current_board = self.session.board

        elif self.controller.board is not None:
            self.current_board = self.controller.board

        elif self.session is not None and self.session.board is not None:
            self.current_board = self.session.board

        if self.session is not None:
            self.last_move = self.session.last_move
            self.ply = self.session.ply

    def start_new_game(self) -> None:
        self.app_mode = AppMode.PLAYING
        self.post_game_overlay_visible = True
        self.concede_requested = False
        self.conceding_color = None
        self.controller = ChessGUIController()
        self.current_board = chess.Board()
        self.last_move = None
        self.ply = 0
        self.renderer.move_tracker_scroll_y = 0
        self.renderer.move_tracker_content_height = 0
        self.renderer.move_tracker_follow_latest = True
        self.renderer.post_game_button_rects = {}

        self.human_agent = HumanGUIAgent(
            controller=self.controller,
            gui_pump=self.pump_once,
            should_quit=self.should_quit,
            should_concede=self.should_concede,
        )

        self.white = (
            self.white_agent_factory()
            if self.white_agent_factory is not None
            else self.human_agent
        )
        self.black = (
            self.black_agent_factory()
            if self.black_agent_factory is not None
            else self._default_black_agent()
        )

        self.white_display_name = (
            self.config.players.white_display_name
            if self.config.players.white_display_name is not None
            else str(self.white)
        )
        self.black_display_name = (
            self.config.players.black_display_name
            if self.config.players.black_display_name is not None
            else str(self.black)
        )

        self.input_adapter = LocalMouseInputAdapter(
            controller=self.controller,
            geometry=self.geometry,
            promotion_menu_geometry=self.promotion_menu_geometry,
            get_view_model=self.get_latest_view_model,
            get_ui_action_at_pixel=self.renderer.ui_action_at_pixel,
        )

        self.controller.set_board(self.current_board)
        self.session = GameSession(
            white=self.white,
            black=self.black,
            controller=self.controller,
            renderer=self.renderer,
            view_model_builder=self.view_model_builder,
            pgn_out=self.pgn_out,
            initial_board=self.current_board,
            white_display_name=self.white_display_name,
            black_display_name=self.black_display_name,
        )

    def _apply_concession_post_game_if_needed(self) -> None:
        if (
            not self.concede_requested
            or self.conceding_color is None
            or self.session is None
        ):
            return

        self.session.apply_concession(
            conceding_color=self.conceding_color,
            white_display_name=self.white_display_name,
            black_display_name=self.black_display_name,
        )

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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def apply_board_perspective(self) -> None:
        perspective = self.config.display.perspective
        human_color = self.config.players.human_color

        if perspective == BoardPerspective.WHITE_BOTTOM:
            self.geometry.set_orientation(True)

        elif perspective == BoardPerspective.PLAYER_BOTTOM:
            self.geometry.set_orientation(human_color == chess.WHITE)

        else:
            raise ValueError(f"Unknown board perspective: {perspective}")


def main() -> None:
    """
    Default executable entry point.

    Human plays White against the default qsearch agent as Black.
    """
    app = ChessGUIApp(
        config=ChessGUIConfig(
            display=DisplayConfig(
                perspective=BoardPerspective.PLAYER_BOTTOM,
            ),
            players=PlayerConfig(
                human_color=chess.WHITE,
            ),
        )
    )
    app.run()


if __name__ == "__main__":
    main()
