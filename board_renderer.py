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
    move_tracker_background: tuple[int, int, int] = (40, 40, 40)
    move_tracker_alt_row_background: tuple[int, int, int] = (48, 48, 48)
    move_tracker_text: tuple[int, int, int] = (235, 235, 235)
    move_tracker_muted_text: tuple[int, int, int] = (170, 170, 170)
    move_tracker_scrollbar: tuple[int, int, int] = (140, 140, 140)
    post_game_overlay: tuple[int, int, int] = (20, 20, 20)
    post_game_panel: tuple[int, int, int] = (40, 40, 40)
    post_game_title: tuple[int, int, int] = (245, 245, 245)
    post_game_body: tuple[int, int, int] = (245, 245, 245)
    post_game_button_top: tuple[int, int, int] = (66, 66, 66)
    post_game_button_bottom: tuple[int, int, int] = (42, 42, 42)
    post_game_button_border: tuple[int, int, int] = (82, 82, 82)
    post_game_button_text: tuple[int, int, int] = (245, 245, 245)
    post_game_close_text: tuple[int, int, int] = (190, 190, 190)


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
        self.move_tracker_gap = 20
        self.move_tracker_min_side_width = 280
        self.move_tracker_max_side_width = 340
        self.move_tracker_min_right_board_size = 480
        self.move_tracker_min_below_height = 120
        self.move_tracker_default_below_height = 180
        self.move_tracker_rect = None
        self.move_tracker_placement = "right"
        self.move_tracker_scroll_y = 0
        self.move_tracker_content_height = 0
        self.move_tracker_reserved_bottom_height = 0
        self.move_tracker_follow_latest = True
        self.post_game_button_rects: dict[str, list[object]] = {}

        self._pygame = self._load_pygame()
        self._coord_font = self._make_coord_font()
        self._player_font = self._make_player_font()
        self._score_font = self._make_score_font()
        self._move_tracker_font = self._make_move_tracker_font()
        self._post_game_title_font = self._make_post_game_title_font()
        self._post_game_body_font = self._make_post_game_body_font()
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

        self.draw_move_tracker(view_model)
        self.draw_post_game(view_model)

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
        self._move_tracker_font = self._make_move_tracker_font()
        self._post_game_title_font = self._make_post_game_title_font()
        self._post_game_body_font = self._make_post_game_body_font()
        self._fallback_piece_font = self._make_fallback_piece_font()

    def resize_to_window(
        self,
        window_width: int,
        window_height: int,
        margin: int = 20,
    ) -> None:
        available_width = window_width - 2 * margin
        available_height = window_height - 2 * margin

        layout = self._calculate_window_layout(
            window_width=window_width,
            window_height=window_height,
            available_width=available_width,
            available_height=available_height,
            margin=margin,
        )

        self.geometry.square_size = max(1, layout["board_size"] // 8)
        self.geometry.board_left = layout["board_left"]
        self.geometry.board_top = layout["board_top"]
        self.move_tracker_rect = layout["move_tracker_rect"]
        self.move_tracker_placement = layout["move_tracker_placement"]
        if self.move_tracker_follow_latest:
            self.move_tracker_scroll_y = self._move_tracker_max_scroll()
        else:
            self._clamp_move_tracker_scroll()
        self.refresh_after_geometry_change()

    def _calculate_window_layout(
        self,
        *,
        window_width: int,
        window_height: int,
        available_width: int,
        available_height: int,
        margin: int,
    ) -> dict[str, object]:
        pygame = self._pygame
        side_width = min(
            self.move_tracker_max_side_width,
            max(self.move_tracker_min_side_width, available_width // 4),
        )
        side_board_space = available_width - side_width - self.move_tracker_gap
        side_stack_height_space = available_height
        side_board_size = min(
            side_board_space,
            side_stack_height_space - 2 * self.panel_height,
        )
        use_right_tracker = (
            side_width >= self.move_tracker_min_side_width
            and side_board_space >= self.move_tracker_min_right_board_size
        )

        if use_right_tracker:
            board_size = max(1, int(side_board_size))
            board_size -= board_size % 8
            board_size = max(8, board_size)
            stack_width = board_size + self.move_tracker_gap + side_width
            stack_height = board_size + 2 * self.panel_height
            stack_left = margin + max(0, available_width - stack_width) // 2
            stack_top = margin + max(0, available_height - stack_height) // 2
            board_left = stack_left
            board_top = stack_top + self.panel_height
            move_tracker_rect = pygame.Rect(
                board_left + board_size + self.move_tracker_gap,
                stack_top,
                side_width,
                stack_height,
            )

            return {
                "board_size": board_size,
                "board_left": board_left,
                "board_top": board_top,
                "move_tracker_rect": move_tracker_rect,
                "move_tracker_placement": "right",
            }

        below_height = min(
            self.move_tracker_default_below_height,
            max(self.move_tracker_min_below_height, available_height // 5),
        )
        below_board_height_space = (
            available_height
            - 2 * self.panel_height
            - self.move_tracker_gap
            - below_height
        )
        board_size = min(available_width, below_board_height_space)
        board_size = max(1, int(board_size))
        board_size -= board_size % 8
        board_size = max(8, board_size)

        stack_height = (
            board_size
            + 2 * self.panel_height
            + self.move_tracker_gap
            + below_height
        )
        stack_top = margin + max(0, available_height - stack_height) // 2
        board_left = (window_width - board_size) // 2
        board_top = stack_top + self.panel_height
        move_tracker_rect = pygame.Rect(
            board_left,
            board_top + board_size + self.panel_height + self.move_tracker_gap,
            board_size,
            below_height,
        )

        return {
            "board_size": board_size,
            "board_left": board_left,
            "board_top": board_top,
            "move_tracker_rect": move_tracker_rect,
            "move_tracker_placement": "below",
        }

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

    def draw_move_tracker(self, view_model: BoardViewModel) -> None:
        if self.move_tracker_rect is None or self.move_tracker_rect.height <= 0:
            return

        pygame = self._pygame
        rect = self.move_tracker_rect
        padding = 10
        row_gap = 4
        row_height = self._move_tracker_font.get_height() + row_gap
        rows = self._move_history_rows(view_model)
        content_height = max(0, len(rows) * row_height - row_gap)
        self.move_tracker_reserved_bottom_height = (
            self._post_game_button_area_height()
            if view_model.post_game is not None
            else 0
        )
        viewport_height = max(
            0,
            rect.height - 2 * padding - self.move_tracker_reserved_bottom_height,
        )
        self.move_tracker_content_height = content_height
        if self.move_tracker_follow_latest:
            self.move_tracker_scroll_y = self._move_tracker_max_scroll()
        else:
            self._clamp_move_tracker_scroll()

        pygame.draw.rect(self.surface, self.colors.move_tracker_background, rect)
        pygame.draw.rect(self.surface, self.colors.border, rect, width=2)

        if not rows or viewport_height <= 0:
            return

        clip_rect = pygame.Rect(
            rect.left + padding,
            rect.top + padding,
            max(0, rect.width - 2 * padding - self._move_tracker_scrollbar_space()),
            viewport_height,
        )
        if clip_rect.width <= 0:
            return

        old_clip = self.surface.get_clip()
        self.surface.set_clip(clip_rect)

        y = clip_rect.top - self.move_tracker_scroll_y
        number_width = self._move_tracker_font.size("99.")[0] + 8
        white_x = clip_rect.left + number_width
        black_x = white_x + max(56, (clip_rect.width - number_width) // 2)

        for row_index, (fullmove_number, white_san, black_san) in enumerate(rows):
            if y + row_height >= clip_rect.top and y <= clip_rect.bottom:
                if row_index % 2 == 1:
                    pygame.draw.rect(
                        self.surface,
                        self.colors.move_tracker_alt_row_background,
                        pygame.Rect(
                            clip_rect.left,
                            y,
                            clip_rect.width,
                            row_height - row_gap,
                        ),
                    )

                number_surface = self._move_tracker_font.render(
                    f"{fullmove_number}.",
                    True,
                    self.colors.move_tracker_muted_text,
                )
                self.surface.blit(number_surface, (clip_rect.left, y))

                if white_san is not None:
                    white_surface = self._move_tracker_font.render(
                        white_san,
                        True,
                        self.colors.move_tracker_text,
                    )
                    self.surface.blit(white_surface, (white_x, y))

                if black_san is not None:
                    black_surface = self._move_tracker_font.render(
                        black_san,
                        True,
                        self.colors.move_tracker_text,
                    )
                    self.surface.blit(black_surface, (black_x, y))

            y += row_height

        self.surface.set_clip(old_clip)
        self._draw_move_tracker_scrollbar(rect, content_height, viewport_height, padding)

    def _move_history_rows(
        self,
        view_model: BoardViewModel,
    ) -> list[tuple[int, Optional[str], Optional[str]]]:
        rows_by_number: dict[int, list[Optional[str]]] = {}

        for entry in view_model.move_history:
            row = rows_by_number.setdefault(entry.fullmove_number, [None, None])
            if entry.color == "white":
                row[0] = entry.san
            else:
                row[1] = entry.san

        return [
            (fullmove_number, row[0], row[1])
            for fullmove_number, row in sorted(rows_by_number.items())
        ]

    def _draw_move_tracker_scrollbar(
        self,
        rect,
        content_height: int,
        viewport_height: int,
        padding: int,
    ) -> None:
        if content_height <= viewport_height or viewport_height <= 0:
            return

        pygame = self._pygame
        track_width = 6
        track_rect = pygame.Rect(
            rect.right - padding - track_width,
            rect.top + padding,
            track_width,
            viewport_height,
        )
        thumb_height = max(18, int(viewport_height * viewport_height / content_height))
        max_scroll = max(1, content_height - viewport_height)
        thumb_travel = max(0, viewport_height - thumb_height)
        thumb_y = track_rect.top + int(
            thumb_travel * self.move_tracker_scroll_y / max_scroll
        )
        thumb_rect = pygame.Rect(
            track_rect.left,
            thumb_y,
            track_width,
            thumb_height,
        )

        pygame.draw.rect(self.surface, self.colors.border, track_rect)
        pygame.draw.rect(self.surface, self.colors.move_tracker_scrollbar, thumb_rect)

    def _move_tracker_scrollbar_space(self) -> int:
        return 14

    def draw_post_game(self, view_model: BoardViewModel) -> None:
        self.post_game_button_rects = {}

        if view_model.post_game is None:
            return

        if view_model.post_game.show_overlay:
            self._draw_post_game_overlay(view_model)

        self._draw_post_game_buttons()

    def _draw_post_game_overlay(self, view_model: BoardViewModel) -> None:
        pygame = self._pygame
        board_rect = pygame.Rect(
            self.geometry.board_left,
            self.geometry.board_top,
            self.geometry.board_size,
            self.geometry.board_size,
        )
        dim_surface = pygame.Surface((board_rect.width, board_rect.height), pygame.SRCALPHA)
        dim_surface.fill((*self.colors.post_game_overlay, 125))
        self.surface.blit(dim_surface, board_rect)

        panel_width = min(
            max(260, int(self.geometry.board_size * 0.72)),
            max(1, self.geometry.board_size - 32),
        )
        panel_height = min(
            max(210, int(self.geometry.board_size * 0.42)),
            max(1, self.geometry.board_size - 32),
        )
        panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        panel_rect.center = board_rect.center

        pygame.draw.rect(
            self.surface,
            self.colors.post_game_panel,
            panel_rect,
            border_radius=12,
        )
        pygame.draw.rect(
            self.surface,
            self.colors.border,
            panel_rect,
            width=2,
            border_radius=12,
        )

        assert view_model.post_game is not None
        close_rect = self._post_game_close_rect(panel_rect)
        self.post_game_button_rects["close_post_game"] = [close_rect]
        close_surface = self._post_game_body_font.render(
            "X",
            True,
            self.colors.post_game_close_text,
        )
        self.surface.blit(close_surface, close_surface.get_rect(center=close_rect.center))

        title_surface = self._post_game_title_font.render(
            view_model.post_game.title,
            True,
            self.colors.post_game_title,
        )
        body_surface = self._post_game_body_font.render(
            view_model.post_game.body,
            True,
            self.colors.post_game_body,
        )
        text_block_top = panel_rect.top + max(42, panel_rect.height // 5)
        title_rect = title_surface.get_rect(
            midtop=(panel_rect.centerx, text_block_top)
        )
        body_rect = body_surface.get_rect(
            midtop=(panel_rect.centerx, title_rect.bottom + 8)
        )
        self.surface.blit(title_surface, title_rect)
        self.surface.blit(body_surface, body_rect)
        self._draw_post_game_popup_buttons(panel_rect)

    def _draw_post_game_buttons(self) -> None:
        button_rects = self._post_game_button_rects()
        self._remember_post_game_button_rects(button_rects)

        for action, rect in button_rects.items():
            self._draw_post_game_button(action, rect)

    def _draw_post_game_popup_buttons(self, panel_rect) -> None:
        button_rects = self._post_game_popup_button_rects(panel_rect)
        self._remember_post_game_button_rects(button_rects)

        for action, rect in button_rects.items():
            self._draw_post_game_button(action, rect)

    def _draw_post_game_button(self, action: str, rect) -> None:
        pygame = self._pygame
        labels = {
            "rematch": "Rematch",
            "quit": "Quit",
        }

        if action not in labels:
            return

        self._draw_rounded_vertical_gradient(
            rect,
            top_color=self.colors.post_game_button_top,
            bottom_color=self.colors.post_game_button_bottom,
            border_color=self.colors.post_game_button_border,
            border_radius=8,
        )

        label_surface = self._post_game_body_font.render(
            labels[action],
            True,
            self.colors.post_game_button_text,
        )
        label_rect = label_surface.get_rect(center=rect.center)
        self.surface.blit(label_surface, label_rect)

    def _draw_rounded_vertical_gradient(
        self,
        rect,
        *,
        top_color: tuple[int, int, int],
        bottom_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
        border_radius: int,
    ) -> None:
        pygame = self._pygame

        if rect.width <= 0 or rect.height <= 0:
            return

        gradient = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        denominator = max(1, rect.height - 1)

        for y in range(rect.height):
            t = y / denominator
            color = tuple(
                int(top_color[index] + (bottom_color[index] - top_color[index]) * t)
                for index in range(3)
            )
            pygame.draw.line(gradient, (*color, 255), (0, y), (rect.width, y))

        mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(
            mask,
            (255, 255, 255, 255),
            mask.get_rect(),
            border_radius=border_radius,
        )
        gradient.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.surface.blit(gradient, rect)
        pygame.draw.rect(
            self.surface,
            border_color,
            rect,
            width=1,
            border_radius=border_radius,
        )

    def _remember_post_game_button_rects(self, button_rects: dict[str, object]) -> None:
        for action, rect in button_rects.items():
            self.post_game_button_rects.setdefault(action, []).append(rect)

    def _post_game_button_rects(self) -> dict[str, object]:
        if self.move_tracker_rect is None:
            return {}

        pygame = self._pygame
        padding = 10
        gap = 8
        button_height = max(34, self._post_game_body_font.get_height() + 12)
        button_width = max(
            88,
            (self.move_tracker_rect.width - 2 * padding - gap) // 2,
        )
        total_width = 2 * button_width + gap
        x = self.move_tracker_rect.left + max(
            padding,
            (self.move_tracker_rect.width - total_width) // 2,
        )
        y = self.move_tracker_rect.bottom - padding - button_height

        return {
            "rematch": pygame.Rect(x, y, button_width, button_height),
            "quit": pygame.Rect(x + button_width + gap, y, button_width, button_height),
        }

    def _post_game_popup_button_rects(self, panel_rect) -> dict[str, object]:
        pygame = self._pygame
        gap = 10
        button_height = max(36, self._post_game_body_font.get_height() + 14)
        button_width = max(100, (panel_rect.width - 48 - gap) // 2)
        total_width = 2 * button_width + gap
        x = panel_rect.centerx - total_width // 2
        y = panel_rect.bottom - button_height - 22

        return {
            "rematch": pygame.Rect(x, y, button_width, button_height),
            "quit": pygame.Rect(x + button_width + gap, y, button_width, button_height),
        }

    def _post_game_close_rect(self, panel_rect):
        pygame = self._pygame
        size = max(28, self._post_game_body_font.get_height() + 8)
        return pygame.Rect(
            panel_rect.right - size - 10,
            panel_rect.top + 10,
            size,
            size,
        )

    def _post_game_button_area_height(self) -> int:
        button_height = max(34, self._post_game_body_font.get_height() + 12)
        return button_height + 20

    def ui_action_at_pixel(self, pos: tuple[int, int]) -> Optional[str]:
        for action, rects in self.post_game_button_rects.items():
            for rect in rects:
                if rect.collidepoint(pos):
                    return action

        return None

    def handle_move_tracker_scroll(
        self,
        scroll_delta_y: int,
        mouse_pos: tuple[int, int],
    ) -> None:
        if self.move_tracker_rect is None:
            return

        if not self.move_tracker_rect.collidepoint(mouse_pos):
            return

        row_height = self._move_tracker_font.get_height() + 4
        self.move_tracker_scroll_y -= scroll_delta_y * row_height * 3
        self._clamp_move_tracker_scroll()
        self.move_tracker_follow_latest = (
            self.move_tracker_scroll_y >= self._move_tracker_max_scroll()
        )

    def _clamp_move_tracker_scroll(self) -> None:
        if self.move_tracker_rect is None:
            self.move_tracker_scroll_y = 0
            return

        max_scroll = self._move_tracker_max_scroll()
        self.move_tracker_scroll_y = max(0, min(self.move_tracker_scroll_y, max_scroll))

    def _move_tracker_max_scroll(self) -> int:
        if self.move_tracker_rect is None:
            return 0

        viewport_height = max(
            0,
            self.move_tracker_rect.height
            - 20
            - self.move_tracker_reserved_bottom_height,
        )
        return max(0, self.move_tracker_content_height - viewport_height)

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

    def _make_move_tracker_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(18, int(self.geometry.square_size * 0.30)))

    def _make_post_game_title_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(24, int(self.geometry.square_size * 0.42)), bold=True)

    def _make_post_game_body_font(self):
        pygame = self._pygame
        return pygame.font.SysFont(None, max(18, int(self.geometry.square_size * 0.28)))

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

    move_tracker_gap = 20
    move_tracker_width = 280
    width = board_left * 2 + geometry.board_size + move_tracker_gap + move_tracker_width
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
    renderer.resize_to_window(width, height)

    return surface, geometry, renderer
