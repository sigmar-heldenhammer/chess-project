# chess_gui_controller.py
"""
Controller layer for a click-based chess GUI.

This module intentionally contains no renderer-specific or input-specific code.
It does not know about pygame, tkinter, PySide, JavaScript, HTML, pixels, or SVG.

Its job is to translate chess-square selections into UI state and completed
python-chess Move objects.

Expected role in the architecture:

    Input Layer
        -> calls controller.handle_square_click(square)

    Controller
        -> tracks selected square / pending move

    Game State
        -> supplied as a python-chess Board

    View Model
        -> can be built from controller.get_ui_state() + board

    Renderer
        -> consumes the view model and draws however it wants

Assumptions about not-yet-implemented pieces:
    1. An input adapter will convert platform-specific input into chess.Square
       values before calling handle_square_click().
       Example:
           pygame mouse pixel -> chess.E2
           browser square id "e2" -> chess.E2

    2. A HumanGUIAgent will call pop_pending_move() while waiting for the user
       to complete a legal move.

    3. A ViewModelBuilder will call get_ui_state() and combine it with the
       current chess.Board to produce renderer-friendly display data.

    4. The external game loop is responsible for actually pushing accepted
       moves onto the board. This controller only proposes a completed move by
       storing it as pending_move.

    5. For the first iteration, promotion is defaulted to queen. Later, this
       can be replaced with a promotion-selection UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import chess


@dataclass(frozen=True)
class UIState:
    """
    UI-only state for the board.

    This is not the chess game state. The actual game state remains the
    python-chess Board.

    selected_square:
        The square currently selected by the user, if any.

    legal_targets:
        Destination squares for the currently selected piece. This is included
        for convenience for the first renderer. It can also be recomputed
        dynamically from board.legal_moves.

    pending_move:
        A completed legal move waiting to be consumed by the HumanGUIAgent.

    message:
        Optional short status/debug message. A renderer may ignore this.
    """

    selected_square: Optional[chess.Square] = None
    legal_targets: tuple[chess.Square, ...] = field(default_factory=tuple)
    legal_captures: tuple[chess.Square, ...] = field(default_factory=tuple)
    pending_move: Optional[chess.Move] = None
    message: Optional[str] = None


class ChessGUIController:
    """
    Platform-independent controller for basic click-to-move behavior.

    This controller accepts chess squares, not pixels and not browser DOM
    events. That keeps it reusable across local GUI and future web frontend.

    Minimal interaction model:
        1. User clicks one of their pieces.
        2. Controller stores that as selected_square.
        3. User clicks a destination square.
        4. If legal, controller stores pending_move.
        5. HumanGUIAgent pops and returns pending_move to the game loop.
    """

    def __init__(self, board: Optional[chess.Board] = None):
        self.board: Optional[chess.Board] = board

        self.selected_square: Optional[chess.Square] = None
        self.pending_move: Optional[chess.Move] = None
        self.message: Optional[str] = None

    # ---------------------------------------------------------------------
    # Board synchronization
    # ---------------------------------------------------------------------

    def set_board(self, board: chess.Board) -> None:
        """
        Attach or update the current board reference.

        Inputs:
            board:
                Current python-chess Board from the game loop.

        Outputs:
            None

        Assumption:
            The game loop or HumanGUIAgent will call this whenever the active
            board changes. In python-chess, Board is mutable, so a persistent
            reference may be sufficient, but explicitly setting it is clearer.
        """
        self.board = board
        self.clear_selection()
        self.message = None

    # ---------------------------------------------------------------------
    # Main controller API
    # ---------------------------------------------------------------------

    def handle_square_click(self, square: chess.Square) -> None:
        """
        Main input entry point.

        Inputs:
            square:
                A python-chess square, such as chess.E2.

        Outputs:
            None

        Behavior:
            - If no square is selected, try to select this square.
            - If a square is already selected, try to complete a move.
            - If the user clicks another own piece, switch selection.
            - If the user clicks the selected square again, clear selection.
        """
        self._require_board()

        if self.selected_square is None:
            self.try_select_square(square)
            return

        if square == self.selected_square:
            self.clear_selection()
            self.message = "Selection cleared."
            return

        # If the user clicks another friendly piece, switch selection instead
        # of attempting an illegal move.
        piece = self.board.piece_at(square)  # type: ignore[union-attr]
        if piece is not None and piece.color == self.board.turn:  # type: ignore[union-attr]
            self.try_select_square(square)
            return

        self.try_complete_move(square)

    def try_select_square(self, square: chess.Square) -> bool:
        """
        Attempt to select a piece.

        Inputs:
            square:
                Candidate selected square.

        Outputs:
            bool:
                True if selection succeeded, False otherwise.

        Selection succeeds only when:
            - a board is available
            - the square contains a piece
            - the piece belongs to the side to move
            - the piece has at least one legal move

        Note:
            The final condition prevents selecting pinned/blocked pieces with
            no legal destinations. You can relax this later if you want the UI
            to allow selecting any own piece.
        """
        self._require_board()
        board = self.board
        assert board is not None

        piece = board.piece_at(square)
        if piece is None:
            self.clear_selection()
            self.message = "No piece on selected square."
            return False

        if piece.color != board.turn:
            self.clear_selection()
            self.message = "Selected piece does not belong to the side to move."
            return False

        legal_targets = self._legal_targets_from(square)

        if not legal_targets:
            self.clear_selection()
            self.message = "Selected piece has no legal moves."
            return False

        self.selected_square = square
        self.message = None
        return True

    def try_complete_move(self, target_square: chess.Square) -> bool:
        """
        Attempt to complete a move from selected_square to target_square.

        Inputs:
            target_square:
                Destination square clicked by the user.

        Outputs:
            bool:
                True if a legal pending_move was created, False otherwise.

        Side effect:
            If legal, self.pending_move is set. The external HumanGUIAgent or
            game loop should consume it using pop_pending_move().
        """
        self._require_board()
        board = self.board
        assert board is not None

        if self.selected_square is None:
            self.message = "No source square selected."
            return False

        candidate_moves = self._candidate_moves(
            from_square=self.selected_square,
            to_square=target_square,
        )

        for move in candidate_moves:
            if move in board.legal_moves:
                self.pending_move = move
                self.clear_selection(keep_message=True)
                self.message = f"Move selected: {move.uci()}"
                return True

        self.clear_selection()
        self.message = "Illegal move."
        return False

    def pop_pending_move(self) -> Optional[chess.Move]:
        """
        Return and clear the currently pending legal move.

        Inputs:
            None

        Outputs:
            chess.Move | None:
                The completed legal move, if one exists.

        Intended caller:
            HumanGUIAgent.select_move().
        """
        move = self.pending_move
        self.pending_move = None
        return move

    # ---------------------------------------------------------------------
    # State exposed to view model builder
    # ---------------------------------------------------------------------

    def get_ui_state(self) -> UIState:
        """
        Return a snapshot of current UI state.

        Inputs:
            None

        Outputs:
            UIState

        Intended caller:
            ViewModelBuilder.

        Note:
            legal_targets is computed fresh from the board and selected_square
            so it cannot become stale after board changes.
        """
        legal_targets: tuple[chess.Square, ...] = tuple()
        legal_captures: tuple[chess.Square, ...] = tuple()


        if self.board is not None and self.selected_square is not None:
            legal_targets = tuple(self._legal_targets_from(self.selected_square))
            legal_captures = tuple(self._legal_captures_from(self.selected_square))


        return UIState(
            selected_square=self.selected_square,
            legal_targets=legal_targets,
            legal_captures=legal_captures,
            pending_move=self.pending_move,
            message=self.message,
        )

    def clear_selection(self, keep_message: bool = False) -> None:
        """
        Clear the currently selected square.

        Inputs:
            keep_message:
                If False, also clear status message.

        Outputs:
            None
        """
        self.selected_square = None

        if not keep_message:
            self.message = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _legal_targets_from(self, square: chess.Square) -> list[chess.Square]:
        """
        Return legal destination squares for all legal moves from a square.
        """
        self._require_board()
        board = self.board
        assert board is not None

        return [
            move.to_square
            for move in board.legal_moves
            if move.from_square == square
        ]

    def _legal_captures_from(self, square: chess.Square) -> list[chess.Square]:
        self._require_board()
        board = self.board
        assert board is not None

        return [
            move.to_square
            for move in board.legal_moves
            if move.from_square == square and board.is_capture(move)
        ]

    def _candidate_moves(
        self,
        from_square: chess.Square,
        to_square: chess.Square,
    ) -> Iterable[chess.Move]:
        """
        Generate candidate moves for a source/destination pair.

        For most moves, there is only one candidate:
            chess.Move(from_square, to_square)

        For promotion moves, python-chess requires specifying the promotion
        piece. For the first iteration, we default to queen.

        Later extension:
            Replace this with a promotion callback or controller state such as:
                self.awaiting_promotion = PromotionRequest(...)
        """
        board = self.board
        assert board is not None

        normal_move = chess.Move(from_square, to_square)

        if self._is_promotion_attempt(from_square, to_square):
            # First-iteration assumption: always promote to queen.
            yield chess.Move(from_square, to_square, promotion=chess.QUEEN)

            # Yielding the normal move afterward is harmless; it will not be
            # legal for actual promotions, but keeps the function generic.
            yield normal_move
        else:
            yield normal_move

    def _is_promotion_attempt(
        self,
        from_square: chess.Square,
        to_square: chess.Square,
    ) -> bool:
        """
        Return True if a move appears to be a pawn promotion attempt.
        """
        board = self.board
        assert board is not None

        piece = board.piece_at(from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False

        target_rank = chess.square_rank(to_square)

        return (
            (piece.color == chess.WHITE and target_rank == 7)
            or
            (piece.color == chess.BLACK and target_rank == 0)
        )

    def _require_board(self) -> None:
        """
        Raise a helpful error if no board has been attached.
        """
        if self.board is None:
            raise RuntimeError(
                "ChessGUIController has no board. "
                "Call controller.set_board(board) before handling input."
            )
