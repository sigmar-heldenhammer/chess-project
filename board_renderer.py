# board_renderer.py
"""
Pygame board renderer for the first local chess GUI.

This renderer consumes BoardViewModel from view_model.py and draws a simple
board with pieces, selected-square highlight, legal-target hints, last-move
highlight, and check highlight.

It intentionally does not know anything about:
    - mouse events
    - agents
    - minimax/search
    - move validation
    - controller internals

Expected architecture:

    Game State + UI State
        ↓
    ViewModelBuilder.build(...)
        ↓
    BoardViewModel
        ↓
    BoardRenderer.draw(view_model)

Assumptions about other not-yet-implemented functionality:
    1. A local app/main loop initializes pygame and creates a display Surface,
       then passes that Surface into BoardRenderer.

    2. BoardGeometry from local_input.py is shared between:
           - LocalMouseInputAdapter
           - BoardRenderer

       This guarantees the renderer and input adapter agree about where squares
       are located.

    3. The first version uses unicode chess piece symbols rendered with a system
       font. Later, this can be replaced with image sprites without changing the
       controller or view model.

    4. BoardViewModel.pieces contains PieceView objects with:
           square: str, e.g. "e4"
           symbol: str, e.g. "P" or "k"
           color: str
           piece_type: str

    5. BoardViewModel selected/target/last/check fields are square names like
       "e4", not python-chess square integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from local_input import BoardGeometry
from view_model import BoardViewModel, PieceView


@dataclass(frozen=True)
class RendererColors:
    """
    Colors used by the local pygame renderer.

    Values are RGB tuples. You can replace this object later to theme the board.
    """

    light_square: tuple[int, int, int] = (238, 238, 210)
    dark_square: tuple[int, int, int] = (118, 150, 86)

    selected: tuple[int, int, int] = (246, 246, 105)
    legal_target: tuple[int, int, int] = (80, 80, 80)
    last_move: tuple[int, int, int] = (186, 202, 68)
    check: tuple[int, int, int] = (220, 70, 70)

    white_piece: tuple[int, int, int] = (245, 245, 245)
    black_piece: tuple[int, int, int] = (20, 20, 20)
    piece_shadow: tuple[int, int, int] = (80, 80, 80)

    background: tuple[int, int, int] = (30, 30, 30)
    border: tuple[int, int, int] = (10, 10, 10)
    message_text: tuple[int, int, int] = (230, 230, 230)


class BoardRenderer:
    """
    Simple pygame renderer for BoardViewModel.

    Public method:
        draw(view_model: BoardViewModel) -> None

    Rendering order:
        1. background
        2. board squares
        3. last move highlight
        4. selected square
        5. check square
        6. legal target hints
        7. pieces
        8. optional message/status
        9. display flip/update
    """

    UNICODE_PIECES = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }

    def __init__(
        self,
        surface,
        geometry: BoardGeometry,
        *,
        colors: Optional[RendererColors] = None,
        show_coordinates: bool = False,
        auto_present: bool = True,
    ):
        """
        Inputs:
            surface:
                pygame display surface or compatible Surface.

            geometry:
                BoardGeometry used to map squares to pixels.

            colors:
                Optional RendererColors palette.

            show_coordinates:
                Whether to draw rank/file coordinates. Defaults to False to
                match the requested minimal first interface.

            auto_present:
                If True, call pygame.display.flip() at the end of draw().
                Set False if a larger app wants to manage presentation itself.

        Outputs:
            BoardRenderer instance.
        """
        self.surface = surface
        self.geometry = geometry
        self.colors = colors or RendererColors()
        self.show_coordinates = show_coordinates
        self.auto_present = auto_present

        self._pygame = self._load_pygame()
        self._piece_font = self._make_piece_font()
        self._coord_font = self._make_coord_font()
        self._message_font = self._make_message_font()

    def draw(self, view_model: BoardViewModel) -> None:
        """
        Draw the complete board view.

        Inputs:
            view_model:
                BoardViewModel produced by ViewModelBuilder.

        Outputs:
            None
        """
        self.clear()
        self.draw_board()

        self.draw_last_move(view_model)
        self.draw_selected_square(view_model)
        self.draw_check_square(view_model)
        self.draw_legal_targets(view_model)

        self.draw_pieces(view_model.pieces)

        if self.show_coordinates:
            self.draw_coordinates()

        self.draw_message(view_model.message)

        if self.auto_present:
            self.present()

    def clear(self) -> None:
        """Fill the whole surface background."""
        self.surface.fill(self.colors.background)

    def draw_board(self) -> None:
        """Draw the 8x8 board squares."""
        pygame = self._pygame

        border_rect = pygame.Rect(
            self.geometry.board_left - 2,
            self.geometry.board_top - 2,
            self.geometry.board_size + 4,
            self.geometry.board_size + 4,
        )
        pygame.draw.rect(self.surface, self.colors.border, border_rect)

        for visual_rank in range(8):
            for visual_file in range(8):
                x = self.geometry.board_left + visual_file * self.geometry.square_size
                y = self.geometry.board_top + visual_rank * self.geometry.square_size

                square = self.geometry.square_from_pixel((x + 1, y + 1))
                if square is None:
                    continue

                file_index = chess.square_file(square)
                rank_index = chess.square_rank(square)
                is_light = (file_index + rank_index) % 2 == 1
                color = self.colors.light_square if is_light else self.colors.dark_square

                rect = pygame.Rect(
                    x,
                    y,
                    self.geometry.square_size,
                    self.geometry.square_size,
                )
                pygame.draw.rect(self.surface, color, rect)

    def draw_last_move(self, view_model: BoardViewModel) -> None:
        """Highlight the origin and destination of the previous move."""
        if view_model.last_move_from is not None:
            self._fill_square_by_name(view_model.last_move_from, self.colors.last_move)

        if view_model.last_move_to is not None:
            self._fill_square_by_name(view_model.last_move_to, self.colors.last_move)

    def draw_selected_square(self, view_model: BoardViewModel) -> None:
        """Highlight currently selected square."""
        if view_model.selected_square is not None:
            self._fill_square_by_name(view_model.selected_square, self.colors.selected)

    def draw_check_square(self, view_model: BoardViewModel) -> None:
        """Highlight checked king square."""
        if view_model.check_square is not None:
            self._fill_square_by_name(view_model.check_square, self.colors.check)

    def draw_legal_targets(self, view_model: BoardViewModel) -> None:
        """Draw simple dots on legal target squares."""
        pygame = self._pygame
        radius = max(5, self.geometry.square_size // 8)

        for square_name in view_model.legal_targets:
            square = chess.parse_square(square_name)
            center = self.geometry.center_pixel_from_square(square)
            pygame.draw.circle(self.surface, self.colors.legal_target, center, radius)

    def draw_pieces(self, pieces: tuple[PieceView, ...]) -> None:
        """Draw all pieces."""
        for piece in pieces:
            self.draw_piece(piece)

    def draw_piece(self, piece: PieceView) -> None:
        """Draw one piece centered on its square."""
        square = chess.parse_square(piece.square)
        center_x, center_y = self.geometry.center_pixel_from_square(square)

        glyph = self.UNICODE_PIECES.get(piece.symbol, piece.symbol)
        piece_color = (
            self.colors.white_piece
            if piece.color == "white"
            else self.colors.black_piece
        )

        shadow_surface = self._piece_font.render(glyph, True, self.colors.piece_shadow)
        shadow_rect = shadow_surface.get_rect(center=(center_x + 2, center_y + 2))
        self.surface.blit(shadow_surface, shadow_rect)

        piece_surface = self._piece_font.render(glyph, True, piece_color)
        piece_rect = piece_surface.get_rect(center=(center_x, center_y))
        self.surface.blit(piece_surface, piece_rect)

    def draw_coordinates(self) -> None:
        """Optionally draw board coordinates."""
        for square in chess.SQUARES:
            name = chess.square_name(square)
            file_char = name[0]
            rank_char = name[1]

            x, y = self.geometry.pixel_from_square(square)

            if (
                self.geometry.white_at_bottom
                and chess.square_rank(square) == 0
            ) or (
                not self.geometry.white_at_bottom
                and chess.square_rank(square) == 7
            ):
                self._draw_small_text(
                    file_char,
                    x + self.geometry.square_size - 12,
                    y + self.geometry.square_size - 16,
                )

            if (
                self.geometry.white_at_bottom
                and chess.square_file(square) == 0
            ) or (
                not self.geometry.white_at_bottom
                and chess.square_file(square) == 7
            ):
                self._draw_small_text(rank_char, x + 4, y + 2)

    def draw_message(self, message: Optional[str]) -> None:
        """
        Draw optional status message below the board.

        The first UI can ignore this visually, but drawing it is helpful during
        early testing when clicks are ignored or illegal.
        """
        if not message:
            return

        text_surface = self._message_font.render(message, True, self.colors.message_text)
        x = self.geometry.board_left
        y = self.geometry.board_top + self.geometry.board_size + 12
        self.surface.blit(text_surface, (x, y))

    def present(self) -> None:
        """Flush the rendered frame to the display."""
        self._pygame.display.flip()

    def _fill_square_by_name(
        self,
        square_name: str,
        color: tuple[int, int, int],
    ) -> None:
        """Fill a single square by algebraic square name."""
        pygame = self._pygame

        square = chess.parse_square(square_name)
        x, y = self.geometry.pixel_from_square(square)

        rect = pygame.Rect(
            x,
            y,
            self.geometry.square_size,
            self.geometry.square_size,
        )
        pygame.draw.rect(self.surface, color, rect)

    def _draw_small_text(self, text: str, x: int, y: int) -> None:
        surface = self._coord_font.render(text, True, self.colors.border)
        self.surface.blit(surface, (x, y))

    def _load_pygame(self):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "BoardRenderer requires pygame. Install pygame to use the "
                "local renderer, or implement a different renderer."
            ) from exc

        if not pygame.font.get_init():
            pygame.font.init()

        return pygame

    def _make_piece_font(self):
        pygame = self._pygame
        font_size = int(self.geometry.square_size * 0.72)

        candidates = [
            "Segoe UI Symbol",
            "DejaVu Sans",
            "Arial Unicode MS",
            "Noto Sans Symbols",
            "FreeSerif",
        ]

        for font_name in candidates:
            path = pygame.font.match_font(font_name)
            if path:
                return pygame.font.Font(path, font_size)

        return pygame.font.SysFont(None, font_size)

    def _make_coord_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(12, self.geometry.square_size // 5))

    def _make_message_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, 24)


def create_pygame_board_window(
    *,
    board_left: int = 20,
    board_top: int = 20,
    square_size: int = 80,
    white_at_bottom: bool = True,
    show_coordinates: bool = False,
):
    """
    Convenience factory for early manual testing.

    Outputs:
        tuple:
            (surface, geometry, renderer)
    """
    import pygame

    pygame.init()

    geometry = BoardGeometry(
        board_left=board_left,
        board_top=board_top,
        square_size=square_size,
        white_at_bottom=white_at_bottom,
    )

    width = board_left * 2 + geometry.board_size
    height = board_top * 2 + geometry.board_size + 40

    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Chess GUI")

    renderer = BoardRenderer(
        surface=surface,
        geometry=geometry,
        show_coordinates=show_coordinates,
    )

    return surface, geometry, renderer
