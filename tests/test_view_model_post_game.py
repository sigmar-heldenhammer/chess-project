import unittest

import chess

from view_model import PostGameView, ViewModelBuilder


class PostGameViewModelTests(unittest.TestCase):
    def setUp(self):
        self.builder = ViewModelBuilder()

    def test_normal_position_has_no_post_game_view(self):
        view_model = self.builder.build(chess.Board())

        self.assertIsNone(view_model.post_game)
        self.assertIsNone(view_model.to_dict()["post_game"])

    def test_post_game_view_serializes_result_and_termination(self):
        post_game = PostGameView(
            result="1-0",
            termination="checkmate",
            title="Game Over: 1-0",
            body="checkmate",
        )

        view_model = self.builder.build(chess.Board(), post_game=post_game)

        self.assertEqual(view_model.post_game, post_game)
        self.assertEqual(
            view_model.to_dict()["post_game"],
            {
                "result": "1-0",
                "termination": "checkmate",
                "title": "Game Over: 1-0",
                "body": "checkmate",
                "show_overlay": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
