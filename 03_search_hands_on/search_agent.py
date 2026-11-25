"""BFS and A* search for a four-connected warehouse grid."""

import heapq
import math
from collections import deque
from pathlib import Path

MOVES = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]


def parse_map(map_text):
    """Parse an ASCII grid and require exactly one start and one goal."""
    if not isinstance(map_text, str):
        raise TypeError("map_text must be a string")
    grid = [line.rstrip("\r") for line in map_text.splitlines() if line != ""]
    if not grid:
        raise ValueError("Map must not be empty")

    allowed = {"#", ".", "S", "G"}
    invalid_symbols = sorted({char for row in grid for char in row if char not in allowed})
    if invalid_symbols:
        raise ValueError(f"Map contains unsupported symbols: {invalid_symbols}")

    rows = len(grid)
    cols = max(len(row) for row in grid)
    if cols == 0:
        raise ValueError("Map rows must not be empty")
    grid = [row.ljust(cols, "#") for row in grid]

    starts = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == "S"]
    goals = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError(
            "Map must contain exactly one S and one G "
            f"(found {len(starts)} S and {len(goals)} G)"
        )
    return grid, rows, cols, starts[0], goals[0]


def in_bounds_and_free(grid, rows, cols, pos):
    row, col = pos
    return 0 <= row < rows and 0 <= col < cols and grid[row][col] != "#"


def neighbors(grid, rows, cols, pos):
    for name, delta_row, delta_col in MOVES:
        nxt = (pos[0] + delta_row, pos[1] + delta_col)
        if in_bounds_and_free(grid, rows, cols, nxt):
            yield name, nxt


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def bfs(grid, rows, cols, start, goal):
    """Return (path, expansion_events); goal is counted when dequeued.

    One expansion event is a non-duplicate state removed from the frontier
    and evaluated. The terminal goal is included; successors are not generated.
    """
    frontier = deque([start])
    visited = {start}
    came_from = {}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == goal:
            return reconstruct_path(came_from, current), expanded

        for _, nxt in neighbors(grid, rows, cols, current):
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            frontier.append(nxt)
    return None, expanded


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zero_heuristic(a, b):
    return 0.0


def astar(grid, rows, cols, start, goal, heuristic=manhattan, weight=1.0):
    """Return (path, expansion_events, cost) with f(n)=g(n)+weight*h(n).

    Improved paths to already-expanded states are reopened. This supports
    inconsistent or weighted heuristics; weighted A* is not guaranteed optimal.
    """
    if weight < 0 or not math.isfinite(weight):
        raise ValueError("heuristic weight must be finite and non-negative")

    counter = 0
    g_score = {start: 0}
    came_from = {}
    frontier = [(weight * heuristic(start, goal), 0, counter, start)]
    expanded_g = {}
    expanded_events = 0

    while frontier:
        _, current_g, _, current = heapq.heappop(frontier)
        if current_g != g_score.get(current):
            continue
        previous_expansion_g = expanded_g.get(current)
        if previous_expansion_g is not None and current_g >= previous_expansion_g:
            continue

        expanded_g[current] = current_g
        expanded_events += 1
        if current == goal:
            return reconstruct_path(came_from, current), expanded_events, current_g

        for _, nxt in neighbors(grid, rows, cols, current):
            tentative_g = current_g + 1
            if tentative_g >= g_score.get(nxt, math.inf):
                continue
            g_score[nxt] = tentative_g
            came_from[nxt] = current
            counter += 1
            f_score = tentative_g + weight * heuristic(nxt, goal)
            heapq.heappush(frontier, (f_score, tentative_g, counter, nxt))

    return None, expanded_events, None


def path_length(path):
    return None if path is None else len(path) - 1


def heuristic_experiments(grid, rows, cols, start, goal):
    """Run the requested h=0, Euclidean, and 2x-Manhattan experiments."""
    configurations = [
        ("Zero heuristic (h=0)", zero_heuristic, 1.0),
        ("Euclidean distance", euclidean, 1.0),
        ("Weighted Manhattan (2 x Manhattan)", manhattan, 2.0),
    ]
    results = []
    for label, heuristic, weight in configurations:
        path, expanded, cost = astar(
            grid, rows, cols, start, goal, heuristic=heuristic, weight=weight
        )
        results.append({
            "heuristic": label,
            "path_length": path_length(path),
            "expanded_states": expanded,
            "path_cost": cost,
            "path": path,
        })
    return results


def report(label, path, expanded, cost=None):
    print(f"--- {label} ---")
    print(f"Solution found: {'Yes' if path is not None else 'No'}")
    if path is not None:
        print(f"Path length (moves): {path_length(path)}")
        if cost is not None:
            print(f"Path cost: {cost}")
        print(f"Path: {path}")
    print(f"States expanded/evaluated: {expanded}")
    print()


def main():
    map_path = Path(__file__).resolve().parent / "warehouse_map.txt"
    map_text = map_path.read_text(encoding="utf-8")
    grid, rows, cols, start, goal = parse_map(map_text)
    print(f"Parsed grid: {rows} rows x {cols} cols, start={start}, goal={goal}\n")

    astar_path, astar_expanded, astar_cost = astar(
        grid, rows, cols, start, goal, heuristic=manhattan
    )
    bfs_path, bfs_expanded = bfs(grid, rows, cols, start, goal)
    report("A* (Manhattan) on original warehouse map", astar_path, astar_expanded, astar_cost)
    report("BFS on original warehouse map", bfs_path, bfs_expanded)

    print("HEURISTIC EXPERIMENTS ON ORIGINAL WAREHOUSE MAP")
    print("=" * 78)
    for result in heuristic_experiments(grid, rows, cols, start, goal):
        print(
            f"{result['heuristic']}: path_length={result['path_length']}, "
            f"expanded_states={result['expanded_states']}, path_cost={result['path_cost']}"
        )
    print(
        "\nExpansion convention: non-stale frontier removals evaluated; goal included; "
        "reopened A* states count again when expanded at a lower g-score."
    )


if __name__ == "__main__":
    main()
