# game_session.py
"""
Game session wrapper for connecting the existing chess engine/game loop to the
new GUI architecture.

The existing simplified interface uses play_game(..., on_update=write_svg),
where on_update receives board/move/ply and writes SVG/FEN files.

This class keeps the same idea, but stores session state and emits GUI updates
instead of writing files.

Expected architecture:

    GameSession.start()
        ↓
    arena.play_game(...)
        ↓
    GameSession.on_position_updated(board, move, ply)
        ↓
    ViewModelBuilder.build(...)
        ↓
    renderer.draw(view_model)

Assumptions about existing project functionality:
    1. arena.play_game exists and has a signature compatible with the current
       play_human.py usage:
           play_game(
               white=...,
               black=...,
               time_control=...,
               on_update=...,
               pgn_out=...
           )

    2. white and black are Agent-like objects with:
           select_move(board: chess.Board, **kwargs) -> chess.Move

    3. The on_update callback is called after moves with:
           board: chess.Board
           move: chess.Move
           ply: int

       This matches the current write_svg(board, move, ply) callback style.

    4. A renderer may be attached now or later. If attached, it exposes:
           draw(view_model) -> None

    5. A controller may be attached now or later. If attached, it exposes:
           get_ui_state() -> UIState

    6. This class does not itself own the local event loop. During human turns,
       HumanGUIAgent.gui_pump is responsible for keeping the GUI responsive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, TextIO

import chess

from view_model import BoardViewModel, PostGameView, ViewModelBuilder


class RendererProtocol(Protocol):
    def draw(self, view_model: BoardViewModel) -> None:
        ...


class ControllerProtocol(Protocol):
    def get_ui_state(self):
        ...

    def set_board(self, board: chess.Board) -> None:
        ...


@dataclass(frozen=True)
class GameSessionState:
    """
    Read-only snapshot of current session state.
    """

    board: chess.Board
    last_move: Optional[chess.Move]
    ply: int
    view_model: BoardViewModel


class GameSession:
    """
    Thin GUI-aware wrapper around the existing arena.play_game function.

    This class should not contain chess-engine logic. It coordinates:
        - players/agents
        - latest board reference
        - last move
        - view model creation
        - optional renderer updates
    """

    def __init__(
        self,
        *,
        white,
        black,
        controller: Optional[ControllerProtocol] = None,
        renderer: Optional[RendererProtocol] = None,
        view_model_builder: Optional[ViewModelBuilder] = None,
        time_control=None,
        pgn_out: Optional[TextIO] = None,
        initial_board: Optional[chess.Board] = None,
        white_display_name: Optional[str] = None,
        black_display_name: Optional[str] = None,
    ):
        """
        Inputs:
            white, black:
                Agent-like players passed to arena.play_game.

            controller:
                Optional ChessGUIController or compatible object.

            renderer:
                Optional renderer exposing draw(view_model).

            view_model_builder:
                Optional custom ViewModelBuilder. Defaults to ViewModelBuilder().

            time_control:
                Passed through to arena.play_game.

            pgn_out:
                Optional file handle passed through to arena.play_game.

            initial_board:
                Optional board for initial rendering before play_game begins.
                Whether this can be passed into play_game depends on your
                existing arena.play_game signature, so this is currently used
                only for GUI state/display.
        """
        self.white = white
        self.black = black
        self.controller = controller
        self.renderer = renderer
        self.view_model_builder = view_model_builder or ViewModelBuilder()
        self.time_control = time_control
        self.pgn_out = pgn_out
        self.white_display_name = white_display_name or str(white)
        self.black_display_name = black_display_name or str(black)

        self.board: chess.Board = initial_board if initial_board is not None else chess.Board()
        self.last_move: Optional[chess.Move] = None
        self.ply: int = 0
        self.final_payload: Optional[dict] = None
        self.post_game: Optional[PostGameView] = None
        self.latest_view_model: BoardViewModel = self._build_view_model()

    def start(self):
        """
        Start the game using the existing arena.play_game function.

        Inputs:
            None

        Outputs:
            Whatever arena.play_game returns, if anything.

        Note:
            We import arena.play_game lazily here so this module can be imported
            in isolation during early GUI development/tests.
        """
        from arena import play_game

        self.render_current_position()

        self.final_payload = play_game(
            white=self.white,
            black=self.black,
            white_name=self.white_display_name,
            black_name=self.black_display_name,
            time_control=self.time_control,
            on_update=self.on_position_updated,
            pgn_out=self.pgn_out,
        )
        self.post_game = self._post_game_from_payload(self.final_payload)
        self.render_current_position()

        return self.final_payload

    def on_position_updated(
        self,
        board: chess.Board,
        move: Optional[chess.Move],
        ply: int,
    ) -> None:
        """
        Callback compatible with the existing play_game(..., on_update=...) hook.

        Inputs:
            board:
                Current python-chess Board after the latest move.

            move:
                Latest move.

            ply:
                Current ply count.

        Outputs:
            None

        Side effects:
            Updates session state and redraws if a renderer is attached.
        """
        self.board = board
        self.last_move = move
        self.ply = ply
        if self.controller is not None:
            self.controller.set_board(board)
        self.latest_view_model = self._build_view_model()

        if self.renderer is not None:
            self.renderer.draw(self.latest_view_model)

    def render_current_position(self) -> BoardViewModel:
        """
        Build and optionally render the current position.

        Inputs:
            None

        Outputs:
            BoardViewModel:
                The latest renderer-facing state.
        """
        self.latest_view_model = self._build_view_model()

        if self.renderer is not None:
            self.renderer.draw(self.latest_view_model)

        return self.latest_view_model

    def get_state(self) -> GameSessionState:
        """
        Return a read-only snapshot of current session state.

        Inputs:
            None

        Outputs:
            GameSessionState
        """
        return GameSessionState(
            board=self.board,
            last_move=self.last_move,
            ply=self.ply,
            view_model=self.latest_view_model,
        )

    def get_view_model(self) -> BoardViewModel:
        """
        Return the latest BoardViewModel.

        Useful for a future web route like:
            GET /state -> session.get_view_model().to_dict()
        """
        return self.latest_view_model

    def _build_view_model(self) -> BoardViewModel:
        """
        Internal helper to combine current board + controller UI state.
        """
        if self.controller is not None:
            return self.view_model_builder.build_from_controller(
                board=self.board,
                controller=self.controller,
                last_move=self.last_move,
                white_display_name=self.white_display_name,
                black_display_name=self.black_display_name,
                post_game=self.post_game,
            )

        return self.view_model_builder.build(
            board=self.board,
            ui_state=None,
            last_move=self.last_move,
            white_display_name=self.white_display_name,
            black_display_name=self.black_display_name,
            post_game=self.post_game,
        )

    def _post_game_from_payload(self, payload: Optional[dict]) -> Optional[PostGameView]:
        if payload is None:
            return None

        result = str(payload.get("result", "*"))
        termination = str(payload.get("termination", "game over"))
        title = self._post_game_title(result)

        return PostGameView(
            result=result,
            termination=termination,
            title=title,
            body=termination,
        )

    def _post_game_title(self, result: str) -> str:
        if result == "1-0":
            return f"{self._truncate_display_name(self.white_display_name)} Won"

        if result == "0-1":
            return f"{self._truncate_display_name(self.black_display_name)} Won"

        if result == "1/2-1/2":
            return "Draw"

        return "Game Over"

    def _truncate_display_name(self, display_name: str) -> str:
        if len(display_name) <= 25:
            return display_name

        return f"{display_name[:22]}..."
