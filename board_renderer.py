# board_renderer.py
"""
Pygame board renderer for the first local chess GUI.

Updated version:
    Uses piece image files from an images/ directory instead of unicode chess
    symbols.

Expected image naming convention:
    images/
        white-pawn.svg
        white-knight.svg
        white-bishop.svg
        white-rook.svg
        white-queen.svg
        white-king.svg
        black-pawn.svg
        black-knight.svg
        black-bishop.svg
        black-rook.svg
        black-queen.svg
        black-king.svg

The extension may be .svg, .png, .jpg, .jpeg, or .webp. SVG is preferred if
you downloaded SVG piece assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess

from local_input import BoardGeometry
from view_model import BoardViewModel, PieceView


@dataclass(frozen=True)
class RendererColors:
    """Colors used by the local pygame renderer."""

    light_square: tuple[int, int, int] = (238, 238, 210)
    dark_square: tuple[int, int, int] = (118, 150, 86)

    selected: tuple[int, int, int] = (246, 246, 105)
    legal_target: tuple[int, int, int] = (80, 80, 80)
    last_move: tuple[int, int, int] = (186, 202, 68)
    check: tuple[int, int, int] = (220, 70, 70)

    background: tuple[int, int, int] = (30, 30, 30)
    border: tuple[int, int, int] = (10, 10, 10)
    message_text: tuple[int, int, int] = (230, 230, 230)
    fallback_piece_text: tuple[int, int, int] = (20, 20, 20)


class PieceImageCache:
    """
    Loads and caches piece images for BoardRenderer.

    Expected filenames:
        {color}-{piece_type}.{ext}

    Examples:
        white-king.svg
        black-rook.png
    """

    SUPPORTED_EXTENSIONS = (".svg", ".png", ".jpg", ".jpeg", ".webp")

    def __init__(self, pygame_module, image_dir: str | Path, target_size: int):
        self.pygame = pygame_module
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self._cache: dict[tuple[str, str, int], object] = {}

    def get(self, color: str, piece_type: str):
        """
        Return a scaled pygame Surface for a piece.

        Raises FileNotFoundError if no matching image exists.
        """
        key = (color, piece_type, self.target_size)

        if key not in self._cache:
            path = self._find_image_path(color, piece_type)
            image = self.pygame.image.load(str(path)).convert_alpha()
            image = self.pygame.transform.smoothscale(
                image,
                (self.target_size, self.target_size),
            )
            self._cache[key] = image

        return self._cache[key]

    def _find_image_path(self, color: str, piece_type: str) -> Path:
        base = f"{color}-{piece_type}"

        for ext in self.SUPPORTED_EXTENSIONS:
            candidate = self.image_dir / f"{base}{ext}"
            if candidate.exists():
                return candidate

        extensionless = self.image_dir / base
        if extensionless.exists():
            return extensionless

        supported = ", ".join(f"{base}{ext}" for ext in self.SUPPORTED_EXTENSIONS)
        raise FileNotFoundError(
            f"Could not find image for {color} {piece_type}. "
            f"Looked in {self.image_dir}. Expected one of: {supported}"
        )


class BoardRenderer:
    """
    Simple pygame renderer for BoardViewModel using image-based pieces.

    It consumes BoardViewModel only; it does not know about input, move
    validation, agents, or controller internals.
    """

    SYMBOL_TO_PIECE_TYPE = {
        "P": "pawn",
        "N": "knight",
        "B": "bishop",
        "R": "rook",
        "Q": "queen",
        "K": "king",
        "p": "pawn",
        "n": "knight",
        "b": "bishop",
        "r": "rook",
        "q": "queen",
        "k": "king",
    }

    def __init__(
        self,
        surface,
        geometry: BoardGeometry,
        *,
        colors: Optional[RendererColors] = None,
        show_coordinates: bool = False,
        auto_present: bool = True,
        image_dir: str | Path = "images",
        piece_scale: float = 0.88,
        allow_missing_piece_fallback: bool = True,
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
                Whether to draw rank/file coordinates.

            auto_present:
                If True, call pygame.display.flip() at the end of draw().

            image_dir:
                Folder containing piece image files.

            piece_scale:
                Fraction of square_size used for piece image width/height.

            allow_missing_piece_fallback:
                If True, missing piece images are drawn as text labels instead
                of crashing. If False, missing images raise FileNotFoundError.
        """
        self.surface = surface
        self.geometry = geometry
        self.colors = colors or RendererColors()
        self.show_coordinates = show_coordinates
        self.auto_present = auto_present
        self.image_dir = self._resolve_image_dir(image_dir)
        self.piece_scale = piece_scale
        self.allow_missing_piece_fallback = allow_missing_piece_fallback

        self._pygame = self._load_pygame()
        self._coord_font = self._make_coord_font()
        self._message_font = self._make_message_font()
        self._fallback_piece_font = self._make_fallback_piece_font()

        piece_image_size = max(1, int(self.geometry.square_size * self.piece_scale))
        self._piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=piece_image_size,
        )

    def draw(self, view_model: BoardViewModel) -> None:
        """Draw the complete board view."""
        self.clear()
        self.draw_board()

        self.draw_last_move(view_model)
        self.draw_selected_square(view_model)
        self.draw_check_square(view_model)
        self.draw_legal_targets(view_model)

        self.draw_pieces(view_model.pieces)
        self.draw_promotion_menu(view_model)

        if self.show_coordinates:
            self.draw_coordinates()

        self.draw_message(view_model.message)

        if self.auto_present:
            self.present()

    def clear(self) -> None:
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


    def draw_promotion_menu(self, view_model: BoardViewModel) -> None:
        if view_model.promotion_request is None:
            return

        pygame = self._pygame

        square = chess.parse_square(view_model.promotion_request.to_square)
        x, y = self.geometry.pixel_from_square(square)

        option_size = self.geometry.square_size

        color = (
            "white"
            if view_model.promotion_request.to_square[1] == "8"
            else "black"
        )

        for index, option in enumerate(view_model.promotion_request.options):
            rect = pygame.Rect(
                x,
                y + index * option_size,
                option_size,
                option_size,
            )

            pygame.draw.rect(self.surface, self.colors.background, rect)
            pygame.draw.rect(self.surface, self.colors.border, rect, width=2)

            image = self._piece_images.get(color, option.piece_type)
            image_rect = image.get_rect(center=rect.center)
            self.surface.blit(image, image_rect)

        # Cancel option
        cancel_index = len(view_model.promotion_request.options)
        cancel_rect = pygame.Rect(
            x,
            y + cancel_index * option_size,
            option_size,
            option_size,
        )

        pygame.draw.rect(self.surface, self.colors.background, cancel_rect)
        pygame.draw.rect(self.surface, self.colors.border, cancel_rect, width=2)

        x_text = self._fallback_piece_font.render("X", True, self.colors.message_text)
        x_rect = x_text.get_rect(center=cancel_rect.center)
        self.surface.blit(x_text, x_rect)

    def refresh_after_geometry_change(self) -> None:
        piece_image_size = max(1, int(self.geometry.square_size * self.piece_scale))

        self._piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=piece_image_size,
        )

        self._coord_font = self._make_coord_font()
        self._message_font = self._make_message_font()
        self._fallback_piece_font = self._make_fallback_piece_font()

    def draw_last_move(self, view_model: BoardViewModel) -> None:
        if view_model.last_move_from is not None:
            self._fill_square_by_name(view_model.last_move_from, self.colors.last_move)

        if view_model.last_move_to is not None:
            self._fill_square_by_name(view_model.last_move_to, self.colors.last_move)

    def draw_selected_square(self, view_model: BoardViewModel) -> None:
        if view_model.selected_square is not None:
            self._fill_square_by_name(view_model.selected_square, self.colors.selected)

    def draw_check_square(self, view_model: BoardViewModel) -> None:
        if view_model.check_square is not None:
            self._fill_square_by_name(view_model.check_square, self.colors.check)

    def draw_legal_targets(self, view_model: BoardViewModel) -> None:
        pygame = self._pygame

        dot_radius = max(5, self.geometry.square_size // 6)
        capture_radius = int(self.geometry.square_size * 0.42)
        capture_width = max(5, self.geometry.square_size // 9)

        legal_captures = set(view_model.legal_captures)

        for square_name in view_model.legal_targets:
            square = chess.parse_square(square_name)
            center = self.geometry.center_pixel_from_square(square)

            if square_name in legal_captures:
                pygame.draw.circle(
                    self.surface,
                    self.colors.legal_target,
                    center,
                    capture_radius,
                    width=capture_width,
                )
            else:
                pygame.draw.circle(
                    self.surface,
                    self.colors.legal_target,
                    center,
                    dot_radius,
                )

    def draw_pieces(self, pieces: tuple[PieceView, ...]) -> None:
        for piece in pieces:
            self.draw_piece(piece)

    def draw_piece(self, piece: PieceView) -> None:
        """Draw one piece centered on its square using image assets."""
        square = chess.parse_square(piece.square)
        center_x, center_y = self.geometry.center_pixel_from_square(square)

        piece_type = piece.piece_type or self.SYMBOL_TO_PIECE_TYPE[piece.symbol]
        color = piece.color

        try:
            image = self._piece_images.get(color, piece_type)
        except FileNotFoundError:
            if not self.allow_missing_piece_fallback:
                raise
            self._draw_missing_piece_fallback(piece, center_x, center_y)
            return

        rect = image.get_rect(center=(center_x, center_y))
        self.surface.blit(image, rect)

    def draw_coordinates(self) -> None:
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
        if not message:
            return

        text_surface = self._message_font.render(message, True, self.colors.message_text)
        x = self.geometry.board_left
        y = self.geometry.board_top + self.geometry.board_size + 12
        self.surface.blit(text_surface, (x, y))

    def present(self) -> None:
        self._pygame.display.flip()

    def _fill_square_by_name(
        self,
        square_name: str,
        color: tuple[int, int, int],
    ) -> None:
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

    def _draw_missing_piece_fallback(
        self,
        piece: PieceView,
        center_x: int,
        center_y: int,
    ) -> None:
        label = piece.symbol
        text_surface = self._fallback_piece_font.render(
            label,
            True,
            self.colors.fallback_piece_text,
        )
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        self.surface.blit(text_surface, text_rect)

    def _resolve_image_dir(self, image_dir: str | Path) -> Path:
        """
        Search order:
            1. Provided path as-is, relative to current working directory.
            2. Path relative to this file's directory.
        """
        raw = Path(image_dir)

        if raw.exists():
            return raw

        relative_to_file = Path(__file__).resolve().parent / raw
        if relative_to_file.exists():
            return relative_to_file

        return raw

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

    def _make_coord_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(12, self.geometry.square_size // 5))

    def _make_message_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, 24)

    def _make_fallback_piece_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, int(self.geometry.square_size * 0.65))


def create_pygame_board_window(
    *,
    board_left: int = 20,
    board_top: int = 20,
    square_size: int = 80,
    white_at_bottom: bool = True,
    show_coordinates: bool = False,
    image_dir: str | Path = "images",
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

    surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Chess GUI")

    renderer = BoardRenderer(
        surface=surface,
        geometry=geometry,
        show_coordinates=show_coordinates,
        image_dir=image_dir,
    )

    return surface, geometry, renderer
