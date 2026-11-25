"""A goal-based warehouse navigation agent using Breadth-First Search (BFS).

The agent keeps the current grid position as its state, searches for a route
from S to G, and returns a sequence of legal Up/Down/Left/Right actions.
All moves have unit cost, so BFS returns a shortest path when one exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

Position = tuple[int, int]
Action = str

# This action order makes equally short solutions deterministic.
MOVES: tuple[tuple[Action, int, int], ...] = (
    ("Up", -1, 0),
    ("Down", 1, 0),
    ("Left", 0, -1),
    ("Right", 0, 1),
)
ALLOWED_SYMBOLS = frozenset("#.SG")

DEFAULT_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
"""


@dataclass(frozen=True)
class Warehouse:
    """Static grid environment and its unique start and goal coordinates."""

    grid: tuple[str, ...]
    start: Position
    goal: Position

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def columns(self) -> int:
        return len(self.grid[0])

    @classmethod
    def from_text(cls, map_text: str) -> "Warehouse":
        """Parse and validate a rectangular ASCII warehouse map.

        Coordinates are zero-based (row, column). Ragged rows are rejected
        rather than silently padded, because padding can change the map.
        """
        if not isinstance(map_text, str):
            raise TypeError("map_text must be a string")

        lines = [line for line in map_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Warehouse map must not be empty")

        width = len(lines[0])
        if width == 0:
            raise ValueError("Warehouse rows must not be empty")
        if any(len(line) != width for line in lines):
            raise ValueError("Warehouse map must be rectangular (all rows equal width)")

        invalid = sorted({cell for line in lines for cell in line if cell not in ALLOWED_SYMBOLS})
        if invalid:
            raise ValueError(f"Unsupported map symbols: {invalid}")

        starts = [(r, c) for r, line in enumerate(lines) for c, cell in enumerate(line) if cell == "S"]
        goals = [(r, c) for r, line in enumerate(lines) for c, cell in enumerate(line) if cell == "G"]
        if len(starts) != 1:
            raise ValueError(f"Map must contain exactly one S (found {len(starts)})")
        if len(goals) != 1:
            raise ValueError(f"Map must contain exactly one G (found {len(goals)})")

        return cls(tuple(lines), starts[0], goals[0])

    def is_free(self, position: Position) -> bool:
        """Return True only for an in-bounds, non-obstacle cell."""
        row, column = position
        return (
            0 <= row < self.rows
            and 0 <= column < self.columns
            and self.grid[row][column] != "#"
        )

    def successors(self, position: Position) -> Iterable[tuple[Action, Position]]:
        """Yield legal one-cell actions in a fixed Up/Down/Left/Right order."""
        for action, dr, dc in MOVES:
            next_position = (position[0] + dr, position[1] + dc)
            if self.is_free(next_position):
                yield action, next_position


class GoalBasedWarehouseAgent:
    """Plan a shortest route to the explicit goal using breadth-first search."""

    def __init__(self, environment: Warehouse) -> None:
        self.environment = environment

    def plan(self) -> Optional[list[Position]]:
        """Return the shortest path including S and G, or None if unreachable.

        BFS stores each discovered state's parent. A state is marked visited
        when it is enqueued, preventing duplicate frontier entries. Once G is
        dequeued, following parent links reconstructs the selected route.
        """
        start = self.environment.start
        goal = self.environment.goal
        frontier: deque[Position] = deque([start])
        visited = {start}
        parent: dict[Position, Optional[Position]] = {start: None}

        while frontier:
            current = frontier.popleft()
            if current == goal:
                return self._reconstruct_path(parent, goal)

            for _action, next_position in self.environment.successors(current):
                if next_position in visited:
                    continue
                visited.add(next_position)
                parent[next_position] = current
                frontier.append(next_position)

        return None

    @staticmethod
    def _reconstruct_path(
        parent: dict[Position, Optional[Position]], goal: Position
    ) -> list[Position]:
        path: list[Position] = []
        current: Optional[Position] = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path


def path_to_actions(path: Optional[list[Position]]) -> Optional[list[Action]]:
    """Convert a coordinate path into named moves; return None for no path."""
    if path is None:
        return None

    delta_to_action = {(dr, dc): action for action, dr, dc in MOVES}
    actions: list[Action] = []
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        try:
            actions.append(delta_to_action[delta])
        except KeyError as exc:
            raise ValueError(f"Path contains a non-adjacent move: {current} -> {nxt}") from exc
    return actions


def render_path(environment: Warehouse, path: Optional[list[Position]]) -> str:
    """Render the selected route with '*' while preserving S, G and walls."""
    if path is None:
        return "\n".join(environment.grid)

    cells = [list(row) for row in environment.grid]
    for row, column in path[1:-1]:
        cells[row][column] = "*"
    return "\n".join("".join(row) for row in cells)


def main() -> int:
    """Load the supplied map, plan, and print a reproducible text result."""
    map_path = Path(__file__).with_name("warehouse_map.txt")
    map_text = map_path.read_text(encoding="utf-8") if map_path.exists() else DEFAULT_MAP

    try:
        environment = Warehouse.from_text(map_text)
    except (TypeError, ValueError) as error:
        print(f"Invalid warehouse map: {error}")
        return 2

    agent = GoalBasedWarehouseAgent(environment)
    path = agent.plan()
    print("GOAL-BASED WAREHOUSE NAVIGATION AGENT")
    print(f"Grid size: {environment.rows} rows x {environment.columns} columns")
    print(f"Start (row, column): {environment.start}")
    print(f"Goal  (row, column): {environment.goal}")

    if path is None:
        print("No collision-free path exists from S to G.")
        return 1

    actions = path_to_actions(path)
    assert actions is not None
    print("Search algorithm: Breadth-First Search (BFS)")
    print(f"Moves required: {len(actions)}")
    print("Action sequence: " + " -> ".join(actions))
    print("Path coordinates (zero-based): " + " -> ".join(map(str, path)))
    print("Route map (* marks the planned path):")
    print(render_path(environment, path))
    print("Result: GOAL REACHED; every move is legal and avoids obstacles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
