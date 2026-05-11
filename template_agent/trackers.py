import uuid
import os

class SearchTracker:
    def __init__(
        self,
        agent_name: str,
        game_id: str,
        output_dir: str,
        agent_color: str | None = None,
    ):
        self.agent_name = agent_name
        self.game_id = game_id
        self.output_dir = output_dir
        self.agent_color = agent_color  # "white" / "black" (optional but recommended)

        # key: (move_number, depth) → count
        self.counts = {}

    def record(self, move_number: int, depth: int):
        key = (move_number, depth)
        self.counts[key] = self.counts.get(key, 0) + 1

    def write_csv(self):
        import csv
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        filename = f"{self.agent_name}_{self.game_id}.csv"
        path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header with metadata fields included
            writer.writerow([
                "agent_name",
                "agent_color",
                "game_id",
                "move_number",
                "depth",
                "search_calls",
            ])

            # Write rows
            for (move_number, depth), count in sorted(self.counts.items()):
                writer.writerow([
                    self.agent_name,
                    self.agent_color,
                    self.game_id,
                    move_number,
                    depth,
                    count,
                ])

def unwrap_agent(agent):
    return getattr(agent, "inner", agent)


def make_tracking_on_root(output_dir: str, enabled: bool = True):
    game_id = uuid.uuid4().hex[:8]
    initialized = False

    def on_root(board, white, black):
        nonlocal initialized

        if initialized or not enabled:
            return

        initialized = True

        for agent, color in [
            (unwrap_agent(white), "white"),
            (unwrap_agent(black), "black"),
        ]:
            agent.tracker = SearchTracker(
                agent_name=getattr(agent, "name", agent.__class__.__name__),
                game_id=game_id,
                output_dir=output_dir,
                agent_color=color,
            )
    return on_root