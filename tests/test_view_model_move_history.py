import unittest

import chess

from view_model import ViewModelBuilder


class MoveHistoryViewModelTests(unittest.TestCase):
    def setUp(self):
        self.builder = ViewModelBuilder()

    def test_starting_position_has_empty_move_history(self):
        view_model = self.builder.build(chess.Board())

        self.assertEqual(view_model.move_history, tuple())
        self.assertEqual(view_model.to_dict()["move_history"], [])

    def test_short_line_records_ply_fullmove_color_san_and_uci(self):
        board = chess.Board()

        for uci in ("e2e4", "e7e5", "g1f3"):
            board.push(chess.Move.from_uci(uci))

        view_model = self.builder.build(board)

        self.assertEqual(
            [
                (
                    entry.ply,
                    entry.fullmove_number,
                    entry.color,
                    entry.san,
                    entry.uci,
                )
                for entry in view_model.move_history
            ],
            [
                (1, 1, "white", "e4", "e2e4"),
                (2, 1, "black", "e5", "e7e5"),
                (3, 2, "white", "Nf3", "g1f3"),
            ],
        )

    def test_special_san_is_computed_by_replay(self):
        board = chess.Board()

        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "e1g1"):
            board.push(chess.Move.from_uci(uci))

        view_model = self.builder.build(board)

        self.assertEqual(view_model.move_history[-1].san, "O-O")

    def test_capture_and_checkmate_san_are_recorded(self):
        board = chess.Board()

        for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
            board.push(chess.Move.from_uci(uci))

        view_model = self.builder.build(board)

        self.assertEqual(view_model.move_history[-1].san, "Qxf7#")

    def test_promotion_san_is_recorded(self):
        board = chess.Board()

        for uci in (
            "a2a4",
            "h7h5",
            "a4a5",
            "h5h4",
            "a5a6",
            "h4h3",
            "a6b7",
            "h3g2",
            "b7a8q",
        ):
            board.push(chess.Move.from_uci(uci))

        view_model = self.builder.build(board)

        self.assertEqual(view_model.move_history[-1].san, "bxa8=Q")

    def test_invalid_move_stack_returns_empty_move_history(self):
        board = chess.Board()
        board.move_stack = [chess.Move.from_uci("a1a8")]

        view_model = self.builder.build(board)

        self.assertEqual(view_model.move_history, tuple())


if __name__ == "__main__":
    unittest.main()
