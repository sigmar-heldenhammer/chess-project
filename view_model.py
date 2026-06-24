# view_model.py
"""
View model layer for the chess GUI.

This module converts the true chess game state plus UI state into a simple,
renderer-friendly data structure.

It does not draw anything and does not handle input.

Expected architecture:

    Game State:    python-chess Board
    UI State:      ChessGUIController.get_ui_state()
        ↓
    ViewModelBuilder.build(...)
        ↓
    BoardViewModel
        ↓
    Renderer.draw(view_model)

The first renderer can be very simple, but this view model is designed to remain
usable when you later replace the local renderer with a web frontend.

Assumptions about other not-yet-implemented functionality:
    1. ChessGUIController exposes get_ui_state(), returning an object with:
           selected_square
           legal_targets
           pending_move
           message

    2. A renderer will consume BoardViewModel and decide how to visually display:
           pieces
           selected square
           legal targets
           last move
           check square

    3. For a future web frontend, BoardViewModel can be serialized to JSON using
       BoardViewModel.to_dict().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import chess


class UIStateProtocol(Protocol):
    selected_square: Optional[chess.Square]
    legal_targets: tuple[chess.Square, ...]
    pending_move: Optional[chess.Move]
    promotion_request: Optional[PromotionRequestProtocol]
    message: Optional[str]


class UIStateProvider(Protocol):
    def get_ui_state(self) -> UIStateProtocol:
        ...


@dataclass(frozen=True)
class PieceView:
    """
    Renderer-friendly representation of a single piece.

    square:
        Square name such as "e4".

    symbol:
        python-chess piece symbol:
            "P", "N", "B", "R", "Q", "K" for white
            "p", "n", "b", "r", "q", "k" for black

    color:
        "white" or "black"

    piece_type:
        Descriptive piece type string.
    """

    square: str
    symbol: str
    color: str
    piece_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "square": self.square,
            "symbol": self.symbol,
            "color": self.color,
            "piece_type": self.piece_type,
        }


@dataclass(frozen=True)
class PromotionOptionView:
    piece_type: str
    piece_type_id: int

@dataclass(frozen=True)
class PromotionRequestView:
    from_square: str
    to_square: str
    options: tuple[PromotionOptionView, ...]


@dataclass(frozen=True)
class CapturedPieceView:
    """
    Renderer-friendly representation of a captured piece icon.

    These icons are normally derived from capture events in board.move_stack.
    They are intentionally display-only; material advantage is computed from
    pieces still on the board so promotions are reflected correctly.
    """

    color: str
    piece_type: str
    symbol: str

    def to_dict(self) -> dict[str, str]:
        return {
            "color": self.color,
            "piece_type": self.piece_type,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class PlayerPanelView:
    color: str
    display_name: str
    captured_pieces: tuple[CapturedPieceView, ...] = field(default_factory=tuple)
    material_advantage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "display_name": self.display_name,
            "captured_pieces": [
                captured_piece.to_dict()
                for captured_piece in self.captured_pieces
            ],
            "material_advantage": self.material_advantage,
        }

@dataclass(frozen=True)
class BoardViewModel:
    """
    Complete renderer-facing state for the board.

    This should contain what the renderer needs to know, but not how to draw it.
    """

    fen: str
    turn: str
    fullmove_number: int
    pieces: tuple[PieceView, ...] = field(default_factory=tuple)

    selected_square: Optional[str] = None
    legal_targets: tuple[str, ...] = field(default_factory=tuple)
    legal_captures: tuple[str, ...] = field(default_factory=tuple)
    last_move: Optional[str] = None
    last_move_from: Optional[str] = None
    last_move_to: Optional[str] = None
    check_square: Optional[str] = None

    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    is_game_over: bool = False
    result: Optional[str] = None

    promotion_request: Optional[PromotionRequestView] = None
    white_panel: PlayerPanelView = field(
        default_factory=lambda: PlayerPanelView(
            color="white",
            display_name="White",
        )
    )
    black_panel: PlayerPanelView = field(
        default_factory=lambda: PlayerPanelView(
            color="black",
            display_name="Black",
        )
    )

    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the view model into JSON-friendly primitives.

        This will be useful for a future web frontend.
        """
        return {
            "fen": self.fen,
            "turn": self.turn,
            "fullmove_number": self.fullmove_number,
            "pieces": [piece.to_dict() for piece in self.pieces],
            "selected_square": self.selected_square,
            "legal_targets": list(self.legal_targets),
            "legal_captures": list(self.legal_captures),
            "last_move": self.last_move,
            "last_move_from": self.last_move_from,
            "last_move_to": self.last_move_to,
            "check_square": self.check_square,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "is_game_over": self.is_game_over,
            "result": self.result,
            "message": self.message,
            "white_panel": self.white_panel.to_dict(),
            "black_panel": self.black_panel.to_dict(),
            "promotion_request": (
                None
                if self.promotion_request is None
                else {
                    "from_square": self.promotion_request.from_square,
                    "to_square": self.promotion_request.to_square,
                    "options": [
                        {
                            "piece_type": option.piece_type,
                            "piece_type_id": option.piece_type_id,
                        }
                        for option in self.promotion_request.options
                    ],
                }
            ),
        }


class PromotionRequestProtocol(Protocol):
    from_square: chess.Square
    to_square: chess.Square
    options: tuple[chess.PieceType, ...]

class ViewModelBuilder:
    """
    Builds BoardViewModel from python-chess board state plus GUI UI state.

    This class is intentionally stateless. It can be reused by a local renderer,
    a web API endpoint, or tests.
    """

    PIECE_TYPE_NAMES = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    PIECE_SYMBOLS = {
        (chess.WHITE, chess.PAWN): "P",
        (chess.WHITE, chess.KNIGHT): "N",
        (chess.WHITE, chess.BISHOP): "B",
        (chess.WHITE, chess.ROOK): "R",
        (chess.WHITE, chess.QUEEN): "Q",
        (chess.BLACK, chess.PAWN): "p",
        (chess.BLACK, chess.KNIGHT): "n",
        (chess.BLACK, chess.BISHOP): "b",
        (chess.BLACK, chess.ROOK): "r",
        (chess.BLACK, chess.QUEEN): "q",
    }
    STARTING_COUNTS = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
    }
    CAPTURE_DISPLAY_ORDER = (
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
        chess.PAWN,
    )
    MATERIAL_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def _build_promotion_request_view(
        self,
        promotion_request: Optional[PromotionRequestProtocol],
    ) -> Optional[PromotionRequestView]:
        if promotion_request is None:
            return None

        return PromotionRequestView(
            from_square=chess.square_name(promotion_request.from_square),
            to_square=chess.square_name(promotion_request.to_square),
            options=tuple(
                PromotionOptionView(
                    piece_type=self.PIECE_TYPE_NAMES[piece_type],
                    piece_type_id=piece_type,
                )
                for piece_type in promotion_request.options
            ),
    )

    def build(
        self,
        board: chess.Board,
        ui_state: Optional[UIStateProtocol] = None,
        *,
        last_move: Optional[chess.Move] = None,
        message: Optional[str] = None,
        white_display_name: str = "White",
        black_display_name: str = "Black",
    ) -> BoardViewModel:
        """
        Build a renderer-friendly view model.

        Inputs:
            board:
                Current python-chess Board.

            ui_state:
                UI-only state from ChessGUIController.get_ui_state().
                May be None for renderers/tests that only need board state.

            last_move:
                Most recent move, if known. This is usually supplied by
                GameSession.on_position_updated(...).

            message:
                Optional app/session-level status message. If omitted, the
                controller ui_state.message is used.

        Outputs:
            BoardViewModel
        """
        pieces = self._build_piece_views(board)

        selected_square = None
        legal_targets: tuple[str, ...] = tuple()
        legal_captures: tuple[str, ...] = tuple()
        ui_message = None
        promotion_request = None

        if ui_state is not None:
            selected_square = self._square_name_or_none(ui_state.selected_square)
            legal_targets = tuple(chess.square_name(sq) for sq in ui_state.legal_targets)
            legal_captures = tuple(chess.square_name(sq) for sq in ui_state.legal_captures)
            ui_message = ui_state.message
            promotion_request = self._build_promotion_request_view(
                ui_state.promotion_request
            )

        last_move_uci = last_move.uci() if last_move is not None else None
        last_move_from = (
            chess.square_name(last_move.from_square) if last_move is not None else None
        )
        last_move_to = (
            chess.square_name(last_move.to_square) if last_move is not None else None
        )

        check_square = self._get_check_square(board)
        white_panel, black_panel = self._build_player_panels(
            board=board,
            white_display_name=white_display_name,
            black_display_name=black_display_name,
        )

        return BoardViewModel(
            fen=board.fen(),
            turn="white" if board.turn == chess.WHITE else "black",
            fullmove_number=board.fullmove_number,
            pieces=pieces,
            selected_square=selected_square,
            legal_targets=legal_targets,
            legal_captures=legal_captures,
            promotion_request=promotion_request,
            last_move=last_move_uci,
            last_move_from=last_move_from,
            last_move_to=last_move_to,
            check_square=check_square,
            is_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            is_game_over=board.is_game_over(),
            result=board.result() if board.is_game_over() else None,
            white_panel=white_panel,
            black_panel=black_panel,
            message=message if message is not None else ui_message,
        )

    def build_from_controller(
        self,
        board: chess.Board,
        controller: UIStateProvider,
        *,
        last_move: Optional[chess.Move] = None,
        message: Optional[str] = None,
        white_display_name: str = "White",
        black_display_name: str = "Black",
    ) -> BoardViewModel:
        """
        Convenience wrapper around build(...).

        Inputs:
            board:
                Current python-chess Board.

            controller:
                Object exposing get_ui_state().

            last_move:
                Most recent move, if known.

            message:
                Optional app/session-level status message.

        Outputs:
            BoardViewModel
        """
        return self.build(
            board=board,
            ui_state=controller.get_ui_state(),
            last_move=last_move,
            message=message,
            white_display_name=white_display_name,
            black_display_name=black_display_name,
        )

    def _build_piece_views(self, board: chess.Board) -> tuple[PieceView, ...]:
        """
        Build a stable list of PieceView objects.

        Sorted by square index so output is deterministic for testing and JSON.
        """
        pieces: list[PieceView] = []

        for square, piece in sorted(board.piece_map().items()):
            piece_type = self.PIECE_TYPE_NAMES[piece.piece_type]
            pieces.append(
                PieceView(
                    square=chess.square_name(square),
                    symbol=piece.symbol(),
                    color="white" if piece.color == chess.WHITE else "black",
                    piece_type=piece_type,
                )
            )

        return tuple(pieces)

    def _build_player_panels(
        self,
        *,
        board: chess.Board,
        white_display_name: str,
        black_display_name: str,
    ) -> tuple[PlayerPanelView, PlayerPanelView]:
        white_material = self._material_on_board(board, chess.WHITE)
        black_material = self._material_on_board(board, chess.BLACK)

        white_panel = PlayerPanelView(
            color="white",
            display_name=white_display_name,
            captured_pieces=self._captured_pieces_by_side(board, chess.WHITE),
            material_advantage=max(0, white_material - black_material),
        )
        black_panel = PlayerPanelView(
            color="black",
            display_name=black_display_name,
            captured_pieces=self._captured_pieces_by_side(board, chess.BLACK),
            material_advantage=max(0, black_material - white_material),
        )

        return white_panel, black_panel

    def _captured_pieces_by_side(
        self,
        board: chess.Board,
        capturing_color: chess.Color,
    ) -> tuple[CapturedPieceView, ...]:
        captured_by_side = self._captured_pieces_from_move_stack(board)

        if captured_by_side is not None:
            return captured_by_side[capturing_color]

        return self._captured_pieces_from_current_board(board, capturing_color)

    def _captured_pieces_from_move_stack(
        self,
        board: chess.Board,
    ) -> Optional[dict[chess.Color, tuple[CapturedPieceView, ...]]]:
        move_stack = getattr(board, "move_stack", None)

        if not move_stack:
            return None

        replay = chess.Board()
        captured_by_side: dict[chess.Color, list[CapturedPieceView]] = {
            chess.WHITE: [],
            chess.BLACK: [],
        }

        try:
            for move in move_stack:
                capturing_color = replay.turn
                captured_piece = self._captured_piece_before_move(replay, move)

                if captured_piece is not None and captured_piece.piece_type != chess.KING:
                    captured_by_side[capturing_color].append(
                        CapturedPieceView(
                            color=(
                                "white"
                                if captured_piece.color == chess.WHITE
                                else "black"
                            ),
                            piece_type=self.PIECE_TYPE_NAMES[captured_piece.piece_type],
                            symbol=captured_piece.symbol(),
                        )
                    )

                replay.push(move)

        except Exception:
            return None

        return {
            chess.WHITE: self._sort_captured_pieces(captured_by_side[chess.WHITE]),
            chess.BLACK: self._sort_captured_pieces(captured_by_side[chess.BLACK]),
        }

    def _captured_piece_before_move(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> Optional[chess.Piece]:
        if board.is_en_passant(move):
            captured_square = chess.square(
                chess.square_file(move.to_square),
                chess.square_rank(move.from_square),
            )
            return board.piece_at(captured_square)

        return board.piece_at(move.to_square)

    def _captured_pieces_from_current_board(
        self,
        board: chess.Board,
        capturing_color: chess.Color,
    ) -> tuple[CapturedPieceView, ...]:
        captured_color = not capturing_color
        captured_pieces: list[CapturedPieceView] = []

        for piece_type in self.CAPTURE_DISPLAY_ORDER:
            current_count = len(board.pieces(piece_type, captured_color))
            missing_count = max(
                0,
                self.STARTING_COUNTS[piece_type] - current_count,
            )

            for _ in range(missing_count):
                captured_pieces.append(
                    CapturedPieceView(
                        color="white" if captured_color == chess.WHITE else "black",
                        piece_type=self.PIECE_TYPE_NAMES[piece_type],
                        symbol=self.PIECE_SYMBOLS[(captured_color, piece_type)],
                    )
                )

        return self._sort_captured_pieces(captured_pieces)

    def _sort_captured_pieces(
        self,
        captured_pieces: list[CapturedPieceView],
    ) -> tuple[CapturedPieceView, ...]:
        order = {
            self.PIECE_TYPE_NAMES[piece_type]: index
            for index, piece_type in enumerate(self.CAPTURE_DISPLAY_ORDER)
        }

        return tuple(
            sorted(
                captured_pieces,
                key=lambda captured_piece: order[captured_piece.piece_type],
            )
        )

    def _material_on_board(self, board: chess.Board, color: chess.Color) -> int:
        total = 0

        for piece_type, value in self.MATERIAL_VALUES.items():
            total += value * len(board.pieces(piece_type, color))

        return total

    def _get_check_square(self, board: chess.Board) -> Optional[str]:
        """
        Return the checked king square, if the side to move is in check.
        """
        if not board.is_check():
            return None

        king_square = board.king(board.turn)
        return self._square_name_or_none(king_square)

    def _square_name_or_none(
        self,
        square: Optional[chess.Square],
    ) -> Optional[str]:
        if square is None:
            return None
        return chess.square_name(square)
