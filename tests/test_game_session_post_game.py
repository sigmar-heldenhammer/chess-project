import unittest

import chess

from game_session import GameSession


class NamedAgent:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class GameSessionPostGameTests(unittest.TestCase):
    def test_white_win_uses_white_display_name(self):
        session = GameSession(
            white=NamedAgent("HumanGUI"),
            black=NamedAgent("Engine"),
        )

        post_game = session._post_game_from_payload(
            {"result": "1-0", "termination": "checkmate"}
        )

        self.assertIsNotNone(post_game)
        self.assertEqual(post_game.title, "HumanGUI Won")
        self.assertEqual(post_game.body, "checkmate")

    def test_black_win_uses_black_display_name(self):
        session = GameSession(
            white=NamedAgent("HumanGUI"),
            black=NamedAgent("Engine"),
        )

        post_game = session._post_game_from_payload(
            {"result": "0-1", "termination": "resignation"}
        )

        self.assertEqual(post_game.title, "Engine Won")

    def test_draw_title_is_draw(self):
        session = GameSession(
            white=NamedAgent("HumanGUI"),
            black=NamedAgent("Engine"),
        )

        post_game = session._post_game_from_payload(
            {"result": "1/2-1/2", "termination": "stalemate"}
        )

        self.assertEqual(post_game.title, "Draw")
        self.assertEqual(post_game.body, "stalemate")

    def test_winner_name_is_truncated_after_twenty_five_characters(self):
        session = GameSession(
            white=NamedAgent("A very very very long display name"),
            black=NamedAgent("Engine"),
        )

        post_game = session._post_game_from_payload(
            {"result": "1-0", "termination": "checkmate"}
        )

        self.assertEqual(post_game.title, "A very very very long ... Won")

    def test_white_concession_awards_win_to_black(self):
        session = GameSession(
            white=NamedAgent("HumanGUI"),
            black=NamedAgent("Engine"),
        )

        session.apply_concession(
            conceding_color=chess.WHITE,
            white_display_name="HumanGUI",
            black_display_name="Engine",
        )

        self.assertEqual(session.post_game.result, "0-1")
        self.assertEqual(session.post_game.title, "Engine Won")
        self.assertEqual(session.post_game.body, "concession")

    def test_black_concession_awards_win_to_white(self):
        session = GameSession(
            white=NamedAgent("HumanGUI"),
            black=NamedAgent("Engine"),
        )

        session.apply_concession(
            conceding_color=chess.BLACK,
            white_display_name="HumanGUI",
            black_display_name="Engine",
        )

        self.assertEqual(session.post_game.result, "1-0")
        self.assertEqual(session.post_game.title, "HumanGUI Won")
        self.assertEqual(session.post_game.body, "concession")


if __name__ == "__main__":
    unittest.main()
