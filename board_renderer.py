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
from io import BytesIO

import chess

from local_input import BoardGeometry
from view_model import BoardViewModel, CapturedPieceView, PieceView, PlayerPanelView


@dataclass(frozen=True)
class RendererColors:
    """Colors used by the local pygame renderer."""

    light_square: tuple[int, int, int] = (238, 238, 210)
    dark_square: tuple[int, int, int] = (118, 150, 86)

    selected: tuple[int, int, int] = (246, 246, 105)
    legal_target_darkening_factor: float = 0.70
    last_move: tuple[int, int, int] = (186, 202, 68)
    check: tuple[int, int, int] = (220, 70, 70)

    background: tuple[int, int, int] = (48, 48, 48)
    border: tuple[int, int, int] = (10, 10, 10)
    promotion_menu_background: tuple[int, int, int] = (245, 245, 245)
    fallback_piece_text: tuple[int, int, int] = (20, 20, 20)
    player_text: tuple[int, int, int] = (235, 235, 235)
    material_advantage: tuple[int, int, int] = (245, 245, 245)


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
        key = (color, piece_type, self.target_size)

        if key not in self._cache:
            path = self._find_image_path(color, piece_type)

            if path.suffix.lower() == ".svg":
                image = self._load_svg_at_target_size(path)
            else:
                image = self.pygame.image.load(str(path)).convert_alpha()
                image = self.pygame.transform.smoothscale(
                    image,
                    (self.target_size, self.target_size),
                )

            self._cache[key] = image

        return self._cache[key]

    def _load_svg_at_target_size(self, path: Path):
        try:
            import cairosvg
        except ImportError as exc:
            raise RuntimeError(
                "SVG piece rendering requires CairoSVG. Install it with:\n"
                "    pip install cairosvg\n"
                "or:\n"
                "    conda install -c conda-forge cairosvg"
            ) from exc

        png_bytes = cairosvg.svg2png(
            url=str(path),
            output_width=self.target_size,
            output_height=self.target_size,
        )

        return self.pygame.image.load(BytesIO(png_bytes), "png").convert_alpha()

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
        promotion_menu_geometry = None,
        panel_height: int = 84,
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
        self.panel_height = panel_height

        self._pygame = self._load_pygame()
        self._coord_font = self._make_coord_font()
        self._player_font = self._make_player_font()
        self._score_font = self._make_score_font()
        self._fallback_piece_font = self._make_fallback_piece_font()
        self.promotion_menu_geometry = promotion_menu_geometry

        piece_image_size = self._board_piece_image_size()
        self._piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=piece_image_size,
        )
        self._captured_piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=self._captured_piece_image_size(),
        )

    def draw(self, view_model: BoardViewModel) -> None:
        """Draw the complete board view."""
        self.clear()
        self.draw_player_panels(view_model)
        self.draw_board()

        self.draw_last_move(view_model)
        self.draw_selected_square(view_model)
        self.draw_check_square(view_model)
        self.draw_legal_targets(view_model)

        self.draw_pieces(view_model.pieces)
        self.draw_promotion_menu(view_model)

        if self.show_coordinates:
            self.draw_coordinates()

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

        if self.promotion_menu_geometry is not None:
            x, y = self.promotion_menu_geometry.menu_origin(view_model.promotion_request)
            option_size = self.promotion_menu_geometry.option_size
        else:
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

            pygame.draw.rect(self.surface, self.colors.promotion_menu_background, rect)
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

        pygame.draw.rect(
            self.surface,
            self.colors.promotion_menu_background,
            cancel_rect,
        )
        pygame.draw.rect(self.surface, self.colors.border, cancel_rect, width=2)

        x_text = self._fallback_piece_font.render("X", True, self.colors.fallback_piece_text)
        x_rect = x_text.get_rect(center=cancel_rect.center)
        self.surface.blit(x_text, x_rect)

    def refresh_after_geometry_change(self) -> None:
        self._piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=self._board_piece_image_size(),
        )
        self._captured_piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=self._captured_piece_image_size(),
        )

        self._coord_font = self._make_coord_font()
        self._player_font = self._make_player_font()
        self._score_font = self._make_score_font()
        self._fallback_piece_font = self._make_fallback_piece_font()

    def resize_to_window(
        self,
        window_width: int,
        window_height: int,
        margin: int = 20,
    ) -> None:
        available_width = window_width - 2 * margin
        available_height = window_height - 2 * margin - 2 * self.panel_height

        board_size = min(available_width, available_height)
        self.geometry.square_size = max(1, board_size // 8)

        self.geometry.board_left = (window_width - self.geometry.board_size) // 2
        self.geometry.board_top = (
            self.panel_height
            + margin
            + max(0, available_height - self.geometry.board_size) // 2
        )
        self.refresh_after_geometry_change()

    def draw_player_panels(self, view_model: BoardViewModel) -> None:
        top_panel = (
            view_model.black_panel
            if self.geometry.white_at_bottom
            else view_model.white_panel
        )
        bottom_panel = (
            view_model.white_panel
            if self.geometry.white_at_bottom
            else view_model.black_panel
        )

        top_rect, bottom_rect = self._player_panel_rects()

        self.draw_player_panel(top_panel, top_rect, vertical_anchor="bottom")
        self.draw_player_panel(bottom_panel, bottom_rect, vertical_anchor="top")

    def _player_panel_rects(self):
        board_top = self.geometry.board_top
        board_bottom = self.geometry.board_top + self.geometry.board_size
        surface_height = self.surface.get_height()

        top_height = min(self.panel_height, max(0, board_top))
        top_rect = self._pygame.Rect(
            self.geometry.board_left,
            board_top - top_height,
            self.geometry.board_size,
            top_height,
        )

        bottom_height = min(
            self.panel_height,
            max(0, surface_height - board_bottom),
        )
        bottom_rect = self._pygame.Rect(
            self.geometry.board_left,
            board_bottom,
            self.geometry.board_size,
            bottom_height,
        )

        return top_rect, bottom_rect

    def draw_player_panel(
        self,
        panel: PlayerPanelView,
        rect,
        *,
        vertical_anchor: str = "top",
    ) -> None:
        if rect.height <= 0:
            return

        name_surface = self._player_font.render(
            panel.display_name,
            True,
            self.colors.player_text,
        )

        padding = 4
        gap = 4
        icon_size = self._panel_captured_piece_image_size(
            rect,
            name_surface,
            panel.captured_pieces,
        )
        self._set_captured_piece_image_size(icon_size)

        content_height = name_surface.get_height() + gap + icon_size
        y = self._player_panel_content_y(
            rect=rect,
            content_height=content_height,
            padding=padding,
            vertical_anchor=vertical_anchor,
        )
        x = rect.left

        self.surface.blit(name_surface, (x, y))

        icon_y = max(
            rect.top + padding,
            min(
                y + name_surface.get_height() + gap,
                rect.bottom - icon_size - padding,
            ),
        )
        next_x = self.draw_captured_pieces(
            panel.captured_pieces,
            x,
            icon_y,
            rect.width,
        )

        if panel.material_advantage > 0:
            score_surface = self._score_font.render(
                f"+{panel.material_advantage}",
                True,
                self.colors.material_advantage,
            )
            score_x = max(
                x,
                min(
                    max(next_x + 8, x),
                    rect.right - score_surface.get_width(),
                ),
            )
            score_y = max(
                rect.top + padding,
                min(
                    icon_y + max(0, (icon_size - score_surface.get_height()) // 2),
                    rect.bottom - score_surface.get_height() - padding,
                ),
            )
            self.surface.blit(score_surface, (score_x, score_y))

    def _player_panel_content_y(
        self,
        *,
        rect,
        content_height: int,
        padding: int,
        vertical_anchor: str,
    ) -> int:
        if vertical_anchor == "bottom":
            return max(
                rect.top + padding,
                rect.bottom - content_height - padding,
            )

        if vertical_anchor == "center":
            return rect.top + max(
                padding,
                (rect.height - content_height) // 2,
            )

        return rect.top + padding

    def draw_captured_pieces(
        self,
        captured_pieces: tuple[CapturedPieceView, ...],
        x: int,
        y: int,
        max_width: int,
    ) -> int:
        if not captured_pieces:
            return x

        icon_size = self._captured_piece_images.target_size
        reserve_for_score = 56
        available = max(0, max_width - reserve_for_score)
        groups = self._group_captured_pieces(captured_pieces)
        group_gap = max(8, int(icon_size * 0.42))
        step = max(4, int(icon_size * 0.58))
        group_count = len(groups)
        repeated_piece_count = len(captured_pieces) - group_count
        needed = (
            group_count * icon_size
            + repeated_piece_count * step
            + max(0, group_count - 1) * group_gap
        )

        if needed > available and group_count > 1:
            group_gap = 4

        if needed > available and repeated_piece_count > 0:
            remaining = available - group_count * icon_size - max(0, group_count - 1) * group_gap
            step = max(1, remaining // repeated_piece_count)

        current_x = x
        for group_index, group in enumerate(groups):
            for captured_piece in group:
                self.draw_captured_piece(captured_piece, current_x, y)
                current_x += step

            current_x += icon_size - step

            if group_index < len(groups) - 1:
                current_x += group_gap

        return current_x

    def draw_captured_piece(
        self,
        captured_piece: CapturedPieceView,
        x: int,
        y: int,
    ) -> None:
        try:
            image = self._captured_piece_images.get(
                captured_piece.color,
                captured_piece.piece_type,
            )
        except FileNotFoundError:
            if not self.allow_missing_piece_fallback:
                raise
            self._draw_captured_piece_fallback(captured_piece, x, y)
            return

        self.surface.blit(image, (x, y))

    def _group_captured_pieces(
        self,
        captured_pieces: tuple[CapturedPieceView, ...],
    ) -> list[list[CapturedPieceView]]:
        groups: list[list[CapturedPieceView]] = []

        for captured_piece in captured_pieces:
            if not groups or groups[-1][0].piece_type != captured_piece.piece_type:
                groups.append([captured_piece])
            else:
                groups[-1].append(captured_piece)

        return groups

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
            indicator_color = self._darkened_surface_color_at(center)

            if square_name in legal_captures:
                pygame.draw.circle(
                    self.surface,
                    indicator_color,
                    center,
                    capture_radius,
                    width=capture_width,
                )
            else:
                pygame.draw.circle(
                    self.surface,
                    indicator_color,
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

    def _darkened_surface_color_at(self, pos: tuple[int, int]) -> tuple[int, int, int]:
        sampled = self.surface.get_at(pos)
        factor = self.colors.legal_target_darkening_factor

        return (
            max(0, min(255, int(sampled.r * factor))),
            max(0, min(255, int(sampled.g * factor))),
            max(0, min(255, int(sampled.b * factor))),
        )

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

    def _draw_captured_piece_fallback(
        self,
        piece: CapturedPieceView,
        x: int,
        y: int,
    ) -> None:
        text_surface = self._fallback_piece_font.render(
            piece.symbol,
            True,
            self.colors.player_text,
        )
        self.surface.blit(text_surface, (x, y))

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

    def _make_player_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(20, self.geometry.square_size // 3))

    def _make_score_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(20, self.geometry.square_size // 3), bold=True)

    def _make_fallback_piece_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, int(self.geometry.square_size * 0.65))

    def _board_piece_image_size(self) -> int:
        return max(1, int(self.geometry.square_size * self.piece_scale))

    def _captured_piece_image_size(self) -> int:
        return max(18, int(self.geometry.square_size * 0.42))

    def _panel_captured_piece_image_size(
        self,
        rect,
        name_surface,
        captured_pieces: tuple[CapturedPieceView, ...],
    ) -> int:
        padding = 4
        gap = 4
        available_height = rect.height - name_surface.get_height() - gap - 2 * padding
        group_count = max(1, len(self._group_captured_pieces(captured_pieces)))
        available_width = max(1, (rect.width - 56) // group_count)
        default_size = self._captured_piece_image_size()

        return max(1, min(default_size, available_height, available_width))

    def _set_captured_piece_image_size(self, target_size: int) -> None:
        if self._captured_piece_images.target_size == target_size:
            return

        self._captured_piece_images = PieceImageCache(
            pygame_module=self._pygame,
            image_dir=self.image_dir,
            target_size=target_size,
        )


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

    panel_height = max(72, int(square_size * 0.95))

    geometry = BoardGeometry(
        board_left=board_left,
        board_top=board_top + panel_height,
        square_size=square_size,
        white_at_bottom=white_at_bottom,
    )

    width = board_left * 2 + geometry.board_size
    height = board_top * 2 + geometry.board_size + panel_height * 2

    surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Chess GUI")

    renderer = BoardRenderer(
        surface=surface,
        geometry=geometry,
        show_coordinates=show_coordinates,
        image_dir=image_dir,
        panel_height=panel_height,
    )

    return surface, geometry, renderer
