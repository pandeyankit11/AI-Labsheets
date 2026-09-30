"""
Independent tests for search_agent.py (Task 3).

A program that produces a plausible path is not necessarily correct, so
each test checks a specific, independently-known expected behaviour
rather than just eyeballing the output.
"""

from search_agent import parse_map, bfs, astar, manhattan, path_length


def load(path):
    with open(path) as f:
        return parse_map(f.read())


def verify_path_is_valid(grid, rows, cols, start, goal, path):
    """Independently re-check a returned path: every step must be a legal
    move (4-connected, in bounds, not an obstacle), it must start at
    start and end at goal, and it must not revisit a cell (no wasted
    steps in a shortest path)."""
    assert path[0] == start, "path does not start at S"
    assert path[-1] == goal, "path does not end at G"
    assert len(set(path)) == len(path), "path revisits a cell"
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        assert abs(r0 - r1) + abs(c0 - c1) == 1, f"illegal move {path[i]} -> {path[i+1]}"
        assert 0 <= r1 < rows and 0 <= c1 < cols, "step leaves the grid"
        assert grid[r1][c1] != "#", "step lands on an obstacle"


def test_1_original_warehouse():
    grid, rows, cols, start, goal = load("warehouse_map.txt")
    path, expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    assert path is not None, "Test 1 FAILED: no path found on the original map"
    verify_path_is_valid(grid, rows, cols, start, goal, path)
    assert path_length(path) == 40, f"Test 1 FAILED: expected 40 moves, got {path_length(path)}"
    print(f"Test 1 PASSED: path length {path_length(path)}, {expanded} states expanded")


def test_2_trivial():
    grid, rows, cols, start, goal = load("map_trivial.txt")
    path, expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    assert path is not None
    verify_path_is_valid(grid, rows, cols, start, goal, path)
    assert path_length(path) == 1, "Test 2 FAILED: expected the one-step solution"
    print(f"Test 2 PASSED: one-step path {path}")


def test_3_no_solution():
    grid, rows, cols, start, goal = load("map_nosolution.txt")
    path, expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    assert path is None, "Test 3 FAILED: planner invented a path to an unreachable goal"
    assert expanded > 0, "Test 3 FAILED: search terminated without exploring anything"
    # sanity: it must not have looped forever -- getting here at all proves termination
    print(f"Test 3 PASSED: correctly reported no path ({expanded} states expanded, search terminated)")


def test_4_alternative_paths_are_shortest():
    grid, rows, cols, start, goal = load("map_altpaths.txt")
    astar_path, astar_expanded, _ = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    bfs_path, bfs_expanded = bfs(grid, rows, cols, start, goal)

    verify_path_is_valid(grid, rows, cols, start, goal, astar_path)
    verify_path_is_valid(grid, rows, cols, start, goal, bfs_path)

    assert path_length(astar_path) == 7, "Test 4 FAILED: A* did not find the shortest (7-move) path"
    assert path_length(bfs_path) == 7, "Test 4 FAILED: BFS did not find a shortest (7-move) path"
    assert astar_expanded <= bfs_expanded, (
        "Test 4 FAILED: A* with an admissible heuristic expanded more states than BFS"
    )
    print(
        f"Test 4 PASSED: both found the 7-move shortest path; "
        f"A* expanded {astar_expanded} states vs BFS's {bfs_expanded}"
    )


if __name__ == "__main__":
    test_1_original_warehouse()
    test_2_trivial()
    test_3_no_solution()
    test_4_alternative_paths_are_shortest()
    print("\nAll independent tests passed.")
