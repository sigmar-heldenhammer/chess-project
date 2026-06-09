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
    last_move: Optional[str] = None
    last_move_from: Optional[str] = None
    last_move_to: Optional[str] = None
    check_square: Optional[str] = None

    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    is_game_over: bool = False
    result: Optional[str] = None

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
        }


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

    def build(
        self,
        board: chess.Board,
        ui_state: Optional[UIStateProtocol] = None,
        *,
        last_move: Optional[chess.Move] = None,
        message: Optional[str] = None,
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
        ui_message = None

        if ui_state is not None:
            selected_square = self._square_name_or_none(ui_state.selected_square)
            legal_targets = tuple(chess.square_name(sq) for sq in ui_state.legal_targets)
            ui_message = ui_state.message

        last_move_uci = last_move.uci() if last_move is not None else None
        last_move_from = (
            chess.square_name(last_move.from_square) if last_move is not None else None
        )
        last_move_to = (
            chess.square_name(last_move.to_square) if last_move is not None else None
        )

        check_square = self._get_check_square(board)

        return BoardViewModel(
            fen=board.fen(),
            turn="white" if board.turn == chess.WHITE else "black",
            fullmove_number=board.fullmove_number,
            pieces=pieces,
            selected_square=selected_square,
            legal_targets=legal_targets,
            last_move=last_move_uci,
            last_move_from=last_move_from,
            last_move_to=last_move_to,
            check_square=check_square,
            is_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            is_game_over=board.is_game_over(),
            result=board.result() if board.is_game_over() else None,
            message=message if message is not None else ui_message,
        )

    def build_from_controller(
        self,
        board: chess.Board,
        controller: UIStateProvider,
        *,
        last_move: Optional[chess.Move] = None,
        message: Optional[str] = None,
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
