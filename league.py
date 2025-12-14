
"""
league.py — Register chess agents and iterate matches to maintain Elo ratings.

CSV schema (created if missing, columns in this exact order):
    AgentID,AgentName,AgentInputs,Elo,Games

Public functions:
    addAgent(agent_name: str, params_str: str, *, csv_path="agents.csv", default_elo=800.0) -> int
    iterateAgents(mode: str, n: int, *, csv_path="agents.csv", k_factor: float = 40.0, tc=None, seed: int | None = None) -> None

Notes:
- Uses a simple registry mapping names to classes; extend AGENT_REGISTRY as you add agents.
- Calls play_game() from arena.py to run games.
- Random color assignment each game.
- Elo updates with standard formula; draw = 0.5. Games increments by 1 per game per agent.
- If an agent cannot be instantiated (unknown class or bad parameters), a warning is printed and that game is skipped.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Import your agents and arena.play_game ---
# Adjust these imports to match your project layout if needed.
try:
    from random_agent import RandomAgent
except Exception:
    RandomAgent = None  # type: ignore

try:
    from greedy_material_agent import GreedyMaterialAgent
except Exception:
    GreedyMaterialAgent = None  # type: ignore

try:
    from minimax_agent import MinimaxAgent
except Exception:
    MinimaxAgent = None  # type: ignore
    
try:
    from evaluation_agent import EvaluationAgent
except Exception:
    EvaluationAgent = None  # type: ignore
    
try:
    from quiescence_agent import QuiescenceAgent
except Exception:
    QuiescenceAgent = None  # type: ignore

try:
    from arena import play_game  # expects signature from previous snippets
except Exception as e:
    raise RuntimeError("Could not import play_game from arena.py. Ensure it exists and is importable.") from e

# ---- Registry mapping agent names to classes ----
AGENT_REGISTRY: Dict[str, Any] = {
    "RandomAgent": RandomAgent,
    "GreedyMaterialAgent": GreedyMaterialAgent,
    "MinimaxAgent": MinimaxAgent,
    "EvaluationAgent": EvaluationAgent,
    "QuiescenceAgent": QuiescenceAgent
}

# ---- CSV constants ----
CSV_COLUMNS = ["AgentID", "AgentName", "AgentInputs", "Elo", "Games"]


# ---- Data helpers ----
@dataclass
class AgentRow:
    AgentID: int
    AgentName: str
    AgentInputs: str
    Elo: float
    Games: int

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "AgentRow":
        return AgentRow(
            AgentID=int(d["AgentID"]),
            AgentName=str(d["AgentName"]).strip(),
            AgentInputs=str(d["AgentInputs"]).strip(),
            Elo=float(d["Elo"]),
            Games=int(d["Games"]),
        )

    def to_dict(self) -> Dict[str, str | int | float]:
        return {
            "AgentID": self.AgentID,
            "AgentName": self.AgentName,
            "AgentInputs": self.AgentInputs,
            "Elo": self.Elo,
            "Games": self.Games,
        }


def _ensure_csv(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def _load_rows(csv_path: str) -> List[AgentRow]:
    _ensure_csv(csv_path)
    rows: List[AgentRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            # tolerate small schema issues (extra spaces, smart quotes in header handled by writer)
            try:
                rows.append(AgentRow.from_dict(r))  # may raise if malformed row
            except Exception as e:
                print(f"[league] Skipping malformed row {r}: {e}")
    return rows


def _save_rows(csv_path: str, rows: List[AgentRow]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())


# ---- Parameter parsing ----
def parse_params(params_str: str) -> Dict[str, Any]:
    """
    Parse 'key=value, key2=value2' into a dict. Supports int, float, bool, and bare strings.
    Trims outer quotes for values. Returns {} for empty/whitespace-only input.
    """
    if params_str is None:
        return {}
    s = params_str.strip()
    if not s:
        return {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: Dict[str, Any] = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Parameter '{p}' is missing '=' (expected key=value).")
        k, v = p.split("=", 1)
        key = k.strip()
        val_raw = v.strip()
        # Strip surrounding quotes (single, double, or smart quotes)
        for q in ['"', "'", "“", "”", "‘", "’"]:
            if val_raw.startswith(q) and val_raw.endswith(q) and len(val_raw) >= 2:
                val_raw = val_raw[1:-1].strip()
                break
        # Try to coerce bool/int/float
        low = val_raw.lower()
        if low in ("true", "false"):
            out[key] = (low == "true")
        else:
            try:
                if "." in val_raw:
                    out[key] = float(val_raw)
                else:
                    out[key] = int(val_raw)
            except ValueError:
                out[key] = val_raw  # keep as string
    return out


# ---- Instantiation ----
def instantiate(agent_name: str, params_str: str):
    cls = AGENT_REGISTRY.get(agent_name)
    if cls is None:
        raise ValueError(f"Unknown agent class '{agent_name}'. (Not in AGENT_REGISTRY)")
    kwargs = parse_params(params_str)
    try:
        return cls(**kwargs)  # type: ignore[operator]
    except TypeError as e:
        raise ValueError(f"Failed to construct {agent_name} with params '{params_str}': {e}")


# ---- Elo helpers ----
def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_elo(r_a: float, r_b: float, result_a: float, k_factor: float) -> Tuple[float, float]:
    """Return updated (R_A, R_B) given A's result in {1.0, 0.5, 0.0}."""
    e_a = expected_score(r_a, r_b)
    e_b = expected_score(r_b, r_a)
    r_a_new = r_a + k_factor * (result_a - e_a)
    r_b_new = r_b + k_factor * ((1.0 - result_a) - e_b)
    return r_a_new, r_b_new


# ---- Public API ----
def addAgent(agent_name: str, params_str: str, *, csv_path: str = "agents.csv", default_elo: float = 800.0) -> int:
    """Register a new agent row. Creates CSV if missing. Returns assigned AgentID."""
    rows = _load_rows(csv_path)
    next_id = 1 + max((r.AgentID for r in rows), default=0)
    new_row = AgentRow(
        AgentID=next_id,
        AgentName=agent_name,
        AgentInputs=params_str,
        Elo=float(default_elo),
        Games=0,
    )
    rows.append(new_row)
    _save_rows(csv_path, rows)
    print(f"[league] Added AgentID={next_id}: {agent_name}({params_str}) | Elo={default_elo}, Games=0")
    return next_id


def iterateAgents(
    mode: str,
    n: int,
    *,
    csv_path: str = "agents.csv",
    k_factor: float = 40.0,
    tc: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Run matches and update Elo/Games.

    mode:
        - "round_robin": every unordered pair plays n games
        - "sample": repeat n times: pick two distinct agents at random and play one game

    k_factor: Elo K-factor (default 40.0)
    tc: optional time control tuple (base_seconds, inc_seconds) passed to play_game
    seed: RNG seed for reproducibility (pairing and color assignment)
    """
    if n <= 0:
        print("[league] n must be >= 1")
        return

    rng = random.Random(seed)
    rows = _load_rows(csv_path)
    if len(rows) < 2:
        print("[league] Need at least 2 agents to iterate matches.")
        return

    # Helper to persist current rows state
    def persist():
        _save_rows(csv_path, rows)

    # Build a mapping from AgentID to row to update in-place
    by_id: Dict[int, AgentRow] = {r.AgentID: r for r in rows}

    # Build list of valid indices for sampling/pairing
    indices = list(range(len(rows)))

    def play_pair(i: int, j: int) -> None:
        a = rows[i]
        b = rows[j]

        # Instantiate agents; on failure, warn and skip.
        try:
            inst_a = instantiate(a.AgentName, a.AgentInputs)
        except Exception as e:
            print(f"[league] Cannot instantiate AgentID={a.AgentID} ({a.AgentName}): {e}. Skipping this game.")
            return
        try:
            inst_b = instantiate(b.AgentName, b.AgentInputs)
        except Exception as e:
            print(f"[league] Cannot instantiate AgentID={b.AgentID} ({b.AgentName}): {e}. Skipping this game.")
            return

        # Random color assignment
        if rng.random() < 0.5:
            white, black = inst_a, inst_b
            white_id, black_id = a.AgentID, b.AgentID
        else:
            white, black = inst_b, inst_a
            white_id, black_id = b.AgentID, a.AgentID

        # Play one game
        try:
            res = play_game(
                white=white,
                black=black,
                time_control=tc,
                quiet=True,
            )
        except Exception as e:
            print(f"[league] play_game failed for pair (AgentID {a.AgentID} vs {b.AgentID}): {e}. Skipping this game.")
            return

        result_str = res.get("result", "")
        # Map to scores for white/black
        if result_str == "1-0":
            s_white, s_black = 1.0, 0.0
        elif result_str == "0-1":
            s_white, s_black = 0.0, 1.0
        else:
            # treat all other as draw
            s_white, s_black = 0.5, 0.5

        # Update Elo
        row_white = by_id[white_id]
        row_black = by_id[black_id]
        new_white_elo, new_black_elo = update_elo(row_white.Elo, row_black.Elo, s_white, k_factor)
        row_white.Elo = new_white_elo
        row_black.Elo = new_black_elo

        # Increment Games
        row_white.Games += 1
        row_black.Games += 1

        # Persist after each game for crash-resilience
        persist()

        # Log a succinct line
        print(f"[league] {row_white.AgentName}(ID {white_id}, W) vs {row_black.AgentName}(ID {black_id}, B) -> {result_str} | Elo now: {row_white.Elo:.1f}/{row_black.Elo:.1f}")

    if mode.lower() == "round_robin":
        # every unordered pair i<j plays n games
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                for _ in range(n):
                    play_pair(i, j)
    elif mode.lower() == "sample":
        # total of n games; each time sample two distinct agents
        for _ in range(n):
            i, j = rng.sample(indices, 2)
            play_pair(i, j)
    else:
        print(f"[league] Unknown mode '{mode}'. Use 'round_robin' or 'sample'.")
        return

    # Final write (ensures any accidental missed persist is flushed)
    persist()


# ---- Optional convenience CLI ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess agent league manager (CSV-based Elo).")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Add/register a new agent")
    p_add.add_argument("name", help="Agent class name (must be in AGENT_REGISTRY)")
    p_add.add_argument("params", help="Parameter string, e.g. depth=2, seed=2")
    p_add.add_argument("--csv", default="agents.csv", help="CSV path (default agents.csv)")
    p_add.add_argument("--elo", type=float, default=800.0, help="Initial Elo (default 800)")

    p_iter = sub.add_parser("iterate", help="Run matches and update Elo")
    p_iter.add_argument("mode", choices=["round_robin", "sample"], help="Iteration mode")
    p_iter.add_argument("n", type=int, help="Round-robin: games per pair; Sample: total games")
    p_iter.add_argument("--csv", default="agents.csv", help="CSV path (default agents.csv)")
    p_iter.add_argument("--k", type=float, default=40.0, help="Elo K-factor (default 40)")
    p_iter.add_argument("--seed", type=int, default=None, help="Random seed")
    p_iter.add_argument("--tc", type=str, default=None, help="Time control as base,inc (e.g., 10,0)")

    args = parser.parse_args()

    if args.cmd == "add":
        addAgent(args.name, args.params, csv_path=args.csv, default_elo=args.elo)
    elif args.cmd == "iterate":
        if args.tc is not None:
            try:
                base_s, inc_s = args.tc.split(",")
                tc = (float(base_s), float(inc_s))
            except Exception:
                print("[league] Could not parse --tc. Use format base,inc (e.g., 10,0). Running without time control.")
                tc = None
        else:
            tc = None

        iterateAgents(
            mode=args.mode,
            n=args.n,
            csv_path=args.csv,
            k_factor=args.k,
            tc=tc,
            seed=args.seed,
        )
    else:
        parser.print_help()
