# Chess Project Structure Notes

These notes summarize the current project as of static review on 2026-06-23.
They are intended for AI agents or maintainers who need to understand the repo
starting from `play_gui.py`.

## What Happens When `play_gui.py` Runs

The intended user command is:

```powershell
python play_gui.py
```

The `main()` function constructs a `ChessGUIApp` with:

- human player as White
- board perspective set to keep the human player at the bottom
- default Black engine created by `template_agent.make_qsearch_agent(depth=1, min_depth=-1)`
- PGN output path `human_gui_game.pgn`

Then `app.run()` opens the PGN file, renders the initial board, and starts
`GameSession.start()`, which calls `arena.play_game(...)`.

The intended runtime flow is:

```text
play_gui.py
  -> ChessGUIApp.__init__()
    -> ChessGUIController()
    -> ViewModelBuilder()
    -> create_pygame_board_window()
      -> pygame.init()
      -> BoardGeometry()
      -> pygame.display.set_mode(...)
      -> BoardRenderer()
    -> PromotionMenuGeometry()
    -> LocalMouseInputAdapter()
    -> HumanGUIAgent()
    -> template_agent.make_qsearch_agent(...)
    -> GameSession(...)
  -> ChessGUIApp.run()
    -> render initial position
    -> GameSession.start()
      -> arena.play_game(...)
```

During the game:

1. `arena.play_game` owns the authoritative `chess.Board`.
2. On White turns, `HumanGUIAgent.select_move(board.copy(stack=True), ...)` blocks until a human move is available.
3. While waiting, `HumanGUIAgent` repeatedly calls `ChessGUIApp.pump_once()`.
4. `pump_once()` asks `LocalMouseInputAdapter` to poll pygame events.
5. Mouse clicks are converted by `BoardGeometry` from pixels to `chess.Square` values.
6. `ChessGUIController` turns source/destination clicks into a legal pending `chess.Move`.
7. `HumanGUIAgent` pops that pending move and returns it to `arena.play_game`.
8. `arena.play_game` validates the move, pushes it onto the board, and calls `GameSession.on_position_updated(...)`.
9. `GameSession` builds a fresh `BoardViewModel` and asks `BoardRenderer` to redraw.
10. On Black turns, the qsearch/minimax agent searches synchronously and returns a move. The GUI may pause during this search.

At game end, `arena.play_game` writes PGN text to `human_gui_game.pgn`, prints the final board/result, and returns a result dictionary.

## Important Current Caveats

- Static review found a naming typo in `play_gui.py`: `ChessGUIApp.__init__` annotates and defaults against `ChessGUIAppConfig`, but the defined class is `ChessGUIConfig`. The normal `main()` path passes a config explicitly, so `self.config = config or ChessGUIAppConfig()` does not evaluate the bad fallback. Creating `ChessGUIApp()` without a config would raise `NameError`.
- `local_input.py` references `Callable` and `BoardViewModel` in annotations without importing them. Because `from __future__ import annotations` is enabled, this should not break normal runtime unless annotations are evaluated.
- Promotion handling has a split representation: the controller stores raw `chess.PieceType` values, while the view model exposes `PromotionOptionView` objects. The current input adapter uses the view-model representation, which is the expected GUI path.
- The renderer expects pygame, python-chess, and CairoSVG if SVG piece assets are used. The `images/` folder currently contains the expected SVG piece files.
- I could not run `python -m py_compile` in this environment because `python.exe` was inaccessible to the sandbox, so these notes are based on static inspection rather than executed verification.

## Key Top-Level Files

- `play_gui.py`: local GUI entry point and application shell. Wires together controller, input, renderer, human GUI agent, engine agent, and game session.
- `arena.py`: authoritative game loop. Alternates between White and Black agents, validates moves, updates the board, calls optional callbacks, manages clocks, writes PGN, and returns result metadata.
- `agents.py`: base `Agent` interface expected by the game loop.
- `human_gui_agent.py`: adapts GUI click input to the `Agent.select_move(...)` interface. It blocks on a human move while pumping the GUI.
- `chess_gui_controller.py`: platform-independent click-to-move controller. Tracks selected square, legal targets, promotion requests, messages, and pending legal moves.
- `local_input.py`: pygame input adapter and board geometry utilities. Converts mouse/window events into controller calls.
- `board_renderer.py`: pygame renderer. Draws board squares, highlights, legal targets, pieces from `images/`, promotion menu, coordinates, and status messages.
- `view_model.py`: converts a `python-chess` board plus controller UI state into renderer-friendly `BoardViewModel` data.
- `game_session.py`: bridge between GUI state/rendering and `arena.play_game(...)`.
- `play_human.py`: older CLI/SVG human-vs-agent entry point that writes `board.svg` and `board.fen`.
- `human_cli_agent.py`: text-input human agent used by `play_human.py`.
- `league.py`, `match.py`, `run_league_demo.py`, `face_off.py`: match/league orchestration utilities for agent-vs-agent testing.
- `tracking_summary.py`, `divergence_logger.py`, `criteria_tester.py`: analysis or diagnostics helpers.

## `template_agent` Package

The `template_agent` folder contains the modular minimax/search framework used
by the default GUI opponent.

- `template_agent/__init__.py`: exports `ModularMinimaxAgent` and factory functions. Note: `__all__` contains a typo, `made_eval_minimax`, but direct imports of `make_eval_minimax` still work because it is imported into the module namespace.
- `template_agent/presets.py`: factory functions such as `make_basic_minimax`, `make_tt_minimax`, `make_history_minimax`, `make_id_minimax`, `make_eval_minimax`, and `make_qsearch_agent`.
- `template_agent/core.py`: `ModularMinimaxAgent`, the canonical alpha-beta search loop.
- `template_agent/agent_templates.py`: shared data structures, protocols, constants, and search result/context types.
- `template_agent/evals.py`: material and weighted-criteria evaluators.
- `template_agent/criteria.py`: named evaluation criteria registry.
- `template_agent/leaves.py`: leaf policies, including normal depth-zero leaves and negative-depth quiescence leaves.
- `template_agent/move_selection.py`: full-width and quiescent move-selection policies.
- `template_agent/ordering.py`: activity and history move ordering.
- `template_agent/tt.py`: transposition table abstractions and implementations.
- `template_agent/depth.py`: depth adjustment policy.
- `template_agent/search_drivers.py`: root search drivers, including iterative deepening.
- `template_agent/trackers.py`: optional search tracking output.

## Data and Generated Output

- `images/`: SVG piece assets consumed by `BoardRenderer`.
- `human_gui_game.pgn`: default PGN output from `play_gui.py`.
- `human_game.pgn`: PGN output from `play_human.py`.
- `board.svg` and `board.fen`: older SVG/FEN outputs from `play_human.py`.
- `agents.csv`: likely agent/match configuration or tracking data.
- `divergences/` and `quiescence_diffs/`: JSON diagnostic output collections.
- `archive-agents/`: older standalone agent implementations kept for reference.

## Interaction Summary

`arena.play_game` is the center of actual chess state. The GUI does not push
moves directly. Instead, the GUI controller proposes a legal pending move, the
human agent returns it through the same interface as an engine agent, and
`arena.play_game` performs the final validation and board mutation.

The rendering stack is deliberately one-way:

```text
python-chess Board + ChessGUIController UIState
  -> ViewModelBuilder
  -> BoardViewModel
  -> BoardRenderer
```

The input stack is deliberately separate:

```text
pygame events
  -> LocalMouseInputAdapter
  -> BoardGeometry
  -> ChessGUIController
  -> HumanGUIAgent.pop_pending_move()
```

This separation means future agents should usually avoid putting chess rules
inside renderer or input code. Chess rules belong in `python-chess`, the
controller's legal move checks, or the search agents.
