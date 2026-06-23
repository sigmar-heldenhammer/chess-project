# local_input.py
"""
Local input layer for the chess GUI.

This module provides the first local/desktop frontend pieces that plug into
ChessGUIController:

    - BoardGeometry:
        Converts between local pixel coordinates and python-chess squares.

    - LocalMouseInputAdapter:
        Reads local mouse/window events and forwards square-clicks to the
        platform-independent ChessGUIController.

This module intentionally keeps all pygame-specific input handling outside of
ChessGUIController. The controller should continue to know only about chess
concepts such as chess.Square and chess.Move.

Expected architecture:

    pygame event / mouse pixel
        ↓
    LocalMouseInputAdapter
        ↓
    BoardGeometry.square_from_pixel(...)
        ↓
    ChessGUIController.handle_square_click(square)

Assumptions about not-yet-implemented functionality:
    1. A pygame-based app/main loop will create a pygame window and call
       LocalMouseInputAdapter.handle_events() once per frame.

    2. A pygame-based renderer will use the same BoardGeometry instance so
       drawing and input coordinate mapping stay consistent.

    3. The app/main loop owns the "running" flag. This adapter returns an
       InputResult to tell the app whether the user requested quit.

    4. ChessGUIController already exists in chess_gui_controller.py and exposes:
           handle_square_click(square: chess.Square) -> None

    5. Board orientation is included now so the same geometry can later support
       board flipping when the human plays black.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import chess


class SquareClickController(Protocol):
    """
    Minimal protocol expected by LocalMouseInputAdapter.

    This avoids requiring a concrete import of ChessGUIController and keeps this
    file loosely coupled. Any future controller with the same method can be used.
    """

    def handle_square_click(self, square: chess.Square) -> None:
        ...
    
    def handle_promotion_choice(self, piece_type: chess.PieceType) -> bool:
        ...


@dataclass(frozen=True)
class InputResult:
    """
    Result returned by LocalMouseInputAdapter.handle_events().

    quit_requested:
        True if the local window received a quit event.

    square_clicked:
        The chess square clicked this frame, if any. This is mainly useful for
        debugging/logging; the adapter already forwards valid clicks to the
        controller.
    """

    quit_requested: bool = False
    square_clicked: Optional[chess.Square] = None
    window_resized: bool = False
    window_size: Optional[tuple[int, int]] = None

@dataclass
class PromotionMenuGeometry:
    board_geometry: BoardGeometry
    option_size: int = 64
    include_cancel: bool = True

    def option_from_pixel(
        self,
        pos: tuple[int, int],
        promotion_request,
    ) -> Optional[chess.PieceType | str]:
        if promotion_request is None:
            return None

        x, y = pos
        menu_left, menu_top = self.menu_origin(promotion_request)

        options = list(promotion_request.options) + ["cancel"]

        for index, option in enumerate(options):
            rect_left = menu_left
            rect_top = menu_top + index * self.option_size

            if (
                rect_left <= x < rect_left + self.option_size
                and rect_top <= y < rect_top + self.option_size
            ):
                if option == "cancel":
                    return "cancel"
                return option.piece_type_id

        return None

    def option_count(self, promotion_request) -> int:
        return len(promotion_request.options) + (1 if self.include_cancel else 0)

    def menu_height(self, promotion_request) -> int:
        return self.option_count(promotion_request) * self.option_size

    def menu_origin(self, promotion_request) -> tuple[int, int]:
        square = chess.parse_square(promotion_request.to_square)
        square_x, square_y = self.board_geometry.pixel_from_square(square)

        board_top = self.board_geometry.board_top
        board_bottom = self.board_geometry.board_top + self.board_geometry.board_size
        height = self.menu_height(promotion_request)

        # Prefer extending downward when it fits.
        if square_y + height <= board_bottom:
            return square_x, square_y

        # Otherwise extend upward, ending on the promotion square.
        return square_x, square_y + self.option_size - height


        
@dataclass
class BoardGeometry:
    """
    Converts between pixel coordinates and python-chess squares.

    This class belongs to the local frontend because pixels are a local rendering
    concept. A future web frontend can implement its own geometry in JavaScript
    or send square names like "e2" directly.

    Coordinate convention:
        - Pixel origin is the top-left of the board.
        - python-chess squares use rank/file, with A1 = 0.
        - If white_at_bottom=True:
              top-left visual square is a8
              bottom-right visual square is h1
        - If white_at_bottom=False:
              top-left visual square is h1
              bottom-right visual square is a8
    """

    board_left: int = 0
    board_top: int = 0
    square_size: int = 80
    white_at_bottom: bool = True

    @property
    def board_size(self) -> int:
        """
        Inputs:
            None

        Outputs:
            int:
                Full board width/height in pixels.
        """
        return self.square_size * 8

    def contains_pixel(self, pos: tuple[int, int]) -> bool:
        """
        Return whether a pixel coordinate is inside the board.

        Inputs:
            pos:
                Local pixel coordinate, usually from a mouse event.

        Outputs:
            bool
        """
        x, y = pos
        return (
            self.board_left <= x < self.board_left + self.board_size
            and self.board_top <= y < self.board_top + self.board_size
        )

    def square_from_pixel(self, pos: tuple[int, int]) -> Optional[chess.Square]:
        """
        Convert a local pixel coordinate into a python-chess square.

        Inputs:
            pos:
                Pixel coordinate as (x, y).

        Outputs:
            chess.Square | None:
                The clicked square, or None if the click was outside the board.
        """
        if not self.contains_pixel(pos):
            return None

        x, y = pos

        visual_file = (x - self.board_left) // self.square_size
        visual_rank_from_top = (y - self.board_top) // self.square_size

        if self.white_at_bottom:
            file_index = visual_file
            rank_index = 7 - visual_rank_from_top
        else:
            file_index = 7 - visual_file
            rank_index = visual_rank_from_top

        return chess.square(file_index, rank_index)

    def pixel_from_square(self, square: chess.Square) -> tuple[int, int]:
        """
        Convert a python-chess square to the top-left local pixel coordinate.

        Inputs:
            square:
                python-chess square.

        Outputs:
            tuple[int, int]:
                Top-left pixel coordinate of the square.
        """
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        if self.white_at_bottom:
            visual_file = file_index
            visual_rank_from_top = 7 - rank_index
        else:
            visual_file = 7 - file_index
            visual_rank_from_top = rank_index

        x = self.board_left + visual_file * self.square_size
        y = self.board_top + visual_rank_from_top * self.square_size

        return x, y

    def center_pixel_from_square(self, square: chess.Square) -> tuple[int, int]:
        """
        Convert a python-chess square to the center local pixel coordinate.

        Inputs:
            square:
                python-chess square.

        Outputs:
            tuple[int, int]:
                Center pixel coordinate of the square.

        This is useful for later features such as arrows, circles, drag pieces,
        or legal-move indicators.
        """
        x, y = self.pixel_from_square(square)
        half = self.square_size // 2
        return x + half, y + half

    def set_orientation(self, white_at_bottom: bool) -> None:
        """
        Set board orientation.

        Inputs:
            white_at_bottom:
                True for normal white-at-bottom orientation, False for flipped.

        Outputs:
            None
        """
        self.white_at_bottom = white_at_bottom

    def resize_to_window(
        self,
        window_width: int,
        window_height: int,
        margin: int = 20,
        ) -> None:
        available_width = window_width - 2 * margin
        available_height = window_height - 2 * margin

        board_size = min(available_width, available_height)
        self.square_size = max(1, board_size // 8)

        self.board_left = (window_width - self.board_size) // 2
        self.board_top = (window_height - self.board_size) // 2     


class LocalMouseInputAdapter:
    """
    pygame-based local input adapter.

    This adapter converts pygame events into chess-square clicks and forwards
    them to ChessGUIController.handle_square_click(...).

    It should be called by the application loop, roughly:

        input_result = input_adapter.handle_events()
        if input_result.quit_requested:
            running = False

    Important:
        This class imports pygame lazily inside handle_events(). That lets the
        rest of the architecture be imported/tested even before pygame is
        installed.
    """

    def __init__(
        self,
        controller: SquareClickController,
        geometry: BoardGeometry,
        promotion_menu_geometry: Optional[PromotionMenuGeometry] = None,
        get_view_model: Optional[Callable[[], BoardViewModel]] = None,
        *,
        left_click_only: bool = True,
    ):
        """
        Inputs:
            controller:
                Object exposing handle_square_click(square).

            geometry:
                BoardGeometry instance shared with the local renderer.

            left_click_only:
                If True, ignore non-left mouse buttons.

        Outputs:
            LocalMouseInputAdapter instance.
        """
        self.controller = controller
        self.geometry = geometry
        self.left_click_only = left_click_only
        self.promotion_menu_geometry = promotion_menu_geometry
        self.get_view_model = get_view_model
        

    def handle_events(self) -> InputResult:
        """
        Poll local pygame events and forward valid board clicks to controller.

        Inputs:
            None

        Outputs:
            InputResult:
                Indicates whether quit was requested and, for debugging, which
                square was clicked.

        Assumption:
            pygame has already been initialized by the app/main loop.
        """
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "LocalMouseInputAdapter requires pygame. "
                "Install pygame or use a different input adapter."
            ) from exc

        quit_requested = False
        square_clicked: Optional[chess.Square] = None
        window_resized = False
        window_size = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                promotion_active = False

                if self.get_view_model is not None:
                    view_model = self.get_view_model()
                    promotion_active = view_model.promotion_request is not None

                if self.left_click_only and event.button != 1 and not promotion_active:
                    continue

                square_clicked = self.handle_click_position(event.pos)
                # self.controller.handle_square_click(square_clicked)

            if event.type == pygame.VIDEORESIZE:
                window_resized = True
                window_size = event.size
                continue

        return InputResult(
            quit_requested=quit_requested,
            square_clicked=square_clicked,
            window_resized=window_resized,
            window_size=window_size,

        )

    def handle_click_position(self, pos: tuple[int, int]) -> Optional[chess.Square]:
        if self.promotion_menu_geometry is not None and self.get_view_model is not None:
            view_model = self.get_view_model()

            if view_model.promotion_request is not None:
                piece_type = self.promotion_menu_geometry.option_from_pixel(
                    pos,
                    view_model.promotion_request,
                )

                if piece_type == "cancel":
                    self.controller.cancel_promotion_request()
                    return None

                if piece_type is not None:
                    self.controller.handle_promotion_choice(piece_type)
                    return None

                # Clicked outside promotion menu: dismiss it.
                self.controller.cancel_promotion_request()
                return None

        square = self.geometry.square_from_pixel(pos)

        if square is None:
            return None

        self.controller.handle_square_click(square)
        return square
