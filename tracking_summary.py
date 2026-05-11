# tracking_summary.py

import os
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_BASE_DIR = "tracking-outputs"


def select_match_folder(base_dir: str = DEFAULT_BASE_DIR) -> Path:
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Tracking output directory not found: {base_path}")

    folders = [
        path for path in base_path.iterdir()
        if path.is_dir()
    ]

    if not folders:
        raise FileNotFoundError(f"No match folders found in: {base_path}")

    # Newest-looking names first. Works well with timestamped match folders.
    folders.sort(key=lambda p: p.name, reverse=True)

    print("\nAvailable match folders:")
    for i, folder in enumerate(folders, start=1):
        print(f"{i}. {folder.name}")

    while True:
        choice = input("\nSelect match folder number: ").strip()

        try:
            idx = int(choice)
            if 1 <= idx <= len(folders):
                return folders[idx - 1]
        except ValueError:
            pass

        print(f"Please enter a number from 1 to {len(folders)}.")


def load_tracking_files(match_folder: Path) -> pd.DataFrame:
    csv_paths = sorted(match_folder.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV tracking files found in: {match_folder}")

    frames = []

    for path in csv_paths:
        df = pd.read_csv(path)

        required_cols = {
            "agent_name",
            "agent_color",
            "game_id",
            "move_number",
            "depth",
            "search_calls",
        }

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing)}"
            )

        df["source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def compute_branching_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute effective branching factor by comparing search_calls at each depth
    to search_calls at the previous higher depth.

    Example for one move:
        depth 2: 1
        depth 1: 20  -> 20 / 1 = 20
        depth 0: 80  -> 80 / 20 = 4

    The highest depth observed for a move is assigned branching factor 1.0.
    """

    records = []

    group_cols = [
        "source_file",
        "agent_name",
        "agent_color",
        "game_id",
        "move_number",
    ]

    for group_key, group in df.groupby(group_cols):
        group = group.sort_values("depth", ascending=False)

        previous_calls: Optional[float] = None

        for _, row in group.iterrows():
            calls = float(row["search_calls"])
            depth = int(row["depth"])

            if previous_calls is None:
                branching_factor = 1.0
            elif previous_calls == 0:
                branching_factor = None
            else:
                branching_factor = calls / previous_calls

            if branching_factor is not None:
                record = dict(zip(group_cols, group_key))
                record.update({
                    "depth": depth,
                    "search_calls": int(row["search_calls"]),
                    "effective_branching_factor": branching_factor,
                })
                records.append(record)

            previous_calls = calls

    return pd.DataFrame(records)


def summarize_branching_factors(branching_df: pd.DataFrame) -> pd.DataFrame:
    summary_long = (
        branching_df
        .groupby(["depth", "agent_name"], as_index=False)
        .agg(
            avg_effective_branching_factor=(
                "effective_branching_factor",
                "mean",
            )
        )
    )

    summary_wide = summary_long.pivot(
        index="depth",
        columns="agent_name",
        values="avg_effective_branching_factor",
    )

    summary_wide = summary_wide.sort_index(ascending=False)

    # Make depth a normal column again instead of an index
    summary_wide = summary_wide.reset_index()

    # Remove pandas' column index name from display/output
    summary_wide.columns.name = None

    return summary_wide

def print_summary(summary: pd.DataFrame, match_folder: Path) -> None:
    print(f"\nSummary for: {match_folder}")

    display = summary.copy()

    for col in display.columns:
        if col != "depth":
            display[col] = display[col].round(3)

    print("\nAverage effective branching factor by depth:\n")
    print(display.to_string(index=False))

def save_summary(summary: pd.DataFrame, match_folder: Path) -> Path:
    output_path = match_folder / "tracking_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def main():
    match_folder = select_match_folder(DEFAULT_BASE_DIR)

    raw_df = load_tracking_files(match_folder)
    branching_df = compute_branching_factors(raw_df)
    summary = summarize_branching_factors(branching_df)

    print_summary(summary, match_folder)

    output_path = save_summary(summary, match_folder)
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()