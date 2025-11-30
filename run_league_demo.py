
"""
run_league_demo.py — Quick interface to register agents and run round-robin matches.

Usage:
    python run_league_demo.py
"""

import os
import csv
import league  # assumes league.py is importable / in the same directory

CSV_PATH = "agents.csv"

# Agents to ensure exist (AgentName, AgentInputs)
AGENTS = [
    ("MinimaxAgent", "depth=2, seed=2, use_alpha_beta=False"),
    ("GreedyMaterialAgent", "seed=1"),
    ("RandomAgent", "seed=12"),
]

def _existing_rows(csv_path: str):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def ensure_agent(agent_name: str, params: str, *, csv_path: str = CSV_PATH, default_elo: float = 800.0):
    """
    Add the agent if an identical (AgentName, AgentInputs) row isn't already present.
    Returns the AgentID (existing or newly created).
    """
    rows = _existing_rows(csv_path)
    for r in rows:
        if r.get("AgentName", "").strip() == agent_name and r.get("AgentInputs", "").strip() == params:
            print(f"[demo] Found existing: AgentID={r.get('AgentID')} {agent_name}({params})")
            try:
                return int(r.get("AgentID"))
            except Exception:
                return None
    # Not found, add it
    return league.addAgent(agent_name, params, csv_path=csv_path, default_elo=default_elo)

if __name__ == "__main__":
    # Ensure agents exist
    for name, params in AGENTS:
        ensure_agent(name, params, csv_path=CSV_PATH, default_elo=800.0)

    # Run a few round-robin games per pair (n=2 gives each unordered pair two games with random colors)
    print("\n[demo] Running round-robin with n=2 ...\n")
    league.iterateAgents(
        mode="round_robin",
        n=2,
        csv_path=CSV_PATH,
        k_factor=40.0,
        tc=None,
        seed=42,   # deterministic color assignment and sampling, if used
    )

    print("\n[demo] Done. Current standings in agents.csv")
