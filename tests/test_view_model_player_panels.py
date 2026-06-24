import unittest

import chess

from view_model import ViewModelBuilder


class PlayerPanelViewModelTests(unittest.TestCase):
    def setUp(self):
        self.builder = ViewModelBuilder()

    def test_starting_position_has_no_captures_or_material_advantage(self):
        view_model = self.builder.build(
            chess.Board(),
            white_display_name="White Agent",
            black_display_name="Black Agent",
        )

        self.assertEqual(view_model.white_panel.display_name, "White Agent")
        self.assertEqual(view_model.black_panel.display_name, "Black Agent")
        self.assertEqual(view_model.white_panel.captured_pieces, tuple())
        self.assertEqual(view_model.black_panel.captured_pieces, tuple())
        self.assertEqual(view_model.white_panel.material_advantage, 0)
        self.assertEqual(view_model.black_panel.material_advantage, 0)

    def test_capture_icons_and_material_advantage_are_separate(self):
        board = chess.Board()

        for square_name in ("a7", "b7", "c7", "b8", "g8", "c8"):
            board.remove_piece_at(chess.parse_square(square_name))

        for square_name in ("a2", "b2", "a1"):
            board.remove_piece_at(chess.parse_square(square_name))

        view_model = self.builder.build(board)

        white_captures = [
            piece.piece_type
            for piece in view_model.white_panel.captured_pieces
        ]
        black_captures = [
            piece.piece_type
            for piece in view_model.black_panel.captured_pieces
        ]

        self.assertEqual(white_captures, ["bishop", "knight", "knight", "pawn", "pawn", "pawn"])
        self.assertEqual(black_captures, ["rook", "pawn", "pawn"])
        self.assertEqual(view_model.white_panel.material_advantage, 5)
        self.assertEqual(view_model.black_panel.material_advantage, 0)

    def test_equal_material_can_still_show_captured_icons(self):
        board = chess.Board()
        board.remove_piece_at(chess.A2)
        board.remove_piece_at(chess.A7)

        view_model = self.builder.build(board)

        self.assertEqual(
            [piece.piece_type for piece in view_model.white_panel.captured_pieces],
            ["pawn"],
        )
        self.assertEqual(
            [piece.piece_type for piece in view_model.black_panel.captured_pieces],
            ["pawn"],
        )
        self.assertEqual(view_model.white_panel.material_advantage, 0)
        self.assertEqual(view_model.black_panel.material_advantage, 0)

    def test_promotion_changes_material_without_matching_captured_icons(self):
        board = chess.Board()
        board.remove_piece_at(chess.A2)
        board.set_piece_at(chess.A3, chess.Piece(chess.QUEEN, chess.WHITE))

        view_model = self.builder.build(board)

        self.assertEqual(view_model.white_panel.material_advantage, 8)
        self.assertEqual(view_model.black_panel.material_advantage, 0)
        self.assertEqual(
            [piece.piece_type for piece in view_model.black_panel.captured_pieces],
            ["pawn"],
        )

    def test_captured_queen_icon_remains_when_a_new_queen_exists(self):
        board = chess.Board()

        for uci in ("e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d2", "e1d2"):
            board.push(chess.Move.from_uci(uci))

        move_stack = list(board.move_stack)
        board.set_piece_at(chess.A3, chess.Piece(chess.QUEEN, chess.BLACK))
        board.move_stack = move_stack
        view_model = self.builder.build(board)

        self.assertIn(
            "queen",
            [piece.piece_type for piece in view_model.white_panel.captured_pieces],
        )

    def test_en_passant_capture_records_pawn_icon(self):
        board = chess.Board()

        for uci in ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"):
            board.push(chess.Move.from_uci(uci))

        view_model = self.builder.build(board)

        self.assertEqual(
            [piece.piece_type for piece in view_model.white_panel.captured_pieces],
            ["pawn"],
        )

    def test_four_captured_pawns_produce_four_icons(self):
        board = chess.Board()

        for square_name in ("a7", "b7", "c7", "d7"):
            board.remove_piece_at(chess.parse_square(square_name))

        view_model = self.builder.build(board)

        self.assertEqual(
            [piece.piece_type for piece in view_model.white_panel.captured_pieces],
            ["pawn", "pawn", "pawn", "pawn"],
        )


if __name__ == "__main__":
    unittest.main()
