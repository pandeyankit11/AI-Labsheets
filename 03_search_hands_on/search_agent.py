"""
Warehouse robot navigation: BFS and A* search over a 4-connected grid.

Generated with the assistance of an LLM (Claude), from the design in
Task 1 of the lab report. Independently re-tested in test_search.py.
"""

import heapq
import math
from collections import deque

# Action name -> (row delta, col delta). Every move has cost 1.
MOVES = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]


def parse_map(map_text):
    """Turn an ASCII map into a grid of rows, plus the S and G coordinates."""
    grid = [line for line in map_text.rstrip("\n").split("\n") if line != ""]
    rows = len(grid)
    cols = max(len(r) for r in grid)
    grid = [r.ljust(cols, "#") for r in grid]  # pad ragged rows with obstacle

    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Map must contain exactly one S and one G")
    return grid, rows, cols, start, goal


def in_bounds_and_free(grid, rows, cols, pos):
    r, c = pos
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    return grid[r][c] != "#"


def neighbors(grid, rows, cols, pos):
    """Yield (action_name, next_state) for every VALID action from pos.
    An action is invalid if it leaves the grid or lands on an obstacle."""
    for name, dr, dc in MOVES:
        nxt = (pos[0] + dr, pos[1] + dc)
        if in_bounds_and_free(grid, rows, cols, nxt):
            yield name, nxt


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Blind search: BFS
# ---------------------------------------------------------------------------

def bfs(grid, rows, cols, start, goal):
    """Breadth-first search. Returns (path, expanded) or (None, expanded)."""
    if start == goal:
        return [start], 0

    frontier = deque([start])
    visited = {start}
    came_from = {}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        for _, nxt in neighbors(grid, rows, cols, current):
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            if nxt == goal:
                return reconstruct_path(came_from, nxt), expanded + 1
            frontier.append(nxt)
    return None, expanded


# ---------------------------------------------------------------------------
# Informed search: A*
# ---------------------------------------------------------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zero_heuristic(a, b):
    return 0


def astar(grid, rows, cols, start, goal, heuristic=manhattan, weight=1.0):
    """A* search. f(n) = g(n) + weight * h(n).
    Returns (path, expanded, cost) or (None, expanded, None)."""
    counter = 0  # tie-breaker so the heap never compares tuples of states
    g_score = {start: 0}
    came_from = {}
    frontier = [(weight * heuristic(start, goal), counter, start)]
    closed = set()
    expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            return reconstruct_path(came_from, current), expanded, g_score[current]

        for _, nxt in neighbors(grid, rows, cols, current):
            tentative_g = g_score[current] + 1
            if nxt in closed and tentative_g >= g_score.get(nxt, math.inf):
                continue
            if tentative_g < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                counter += 1
                f_score = tentative_g + weight * heuristic(nxt, goal)
                heapq.heappush(frontier, (f_score, counter, nxt))

    return None, expanded, None


def path_length(path):
    """Number of moves (edges), not number of cells."""
    return None if path is None else len(path) - 1


def report(label, path, expanded, cost=None):
    print(f"--- {label} ---")
    if path is None:
        print(f"Solution found: No")
        print(f"States expanded: {expanded}")
    else:
        print(f"Solution found: Yes")
        print(f"Path length (moves): {path_length(path)}")
        if cost is not None:
            print(f"Path cost: {cost}")
        print(f"States expanded: {expanded}")
        print(f"Path: {path}")
    print()


if __name__ == "__main__":
    with open("warehouse_map.txt") as f:
        map_text = f.read()
    grid, rows, cols, start, goal = parse_map(map_text)
    print(f"Parsed grid: {rows} rows x {cols} cols, start={start}, goal={goal}\n")

    path, expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    report("A* (Manhattan) on original warehouse map", path, expanded, cost)

    bfs_path, bfs_expanded = bfs(grid, rows, cols, start, goal)
    report("BFS on original warehouse map", bfs_path, bfs_expanded)
