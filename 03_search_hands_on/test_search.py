"""Independent tests for map parsing, BFS, A*, and heuristic experiments."""

from pathlib import Path

from search_agent import (
    astar, bfs, euclidean, heuristic_experiments, manhattan, parse_map, path_length,
)

BASE = Path(__file__).resolve().parent


def load(filename):
    return parse_map((BASE / filename).read_text(encoding="utf-8"))


def verify_path_is_valid(grid, rows, cols, start, goal, path):
    assert path is not None, "expected a path"
    assert path[0] == start, "path does not start at S"
    assert path[-1] == goal, "path does not end at G"
    assert len(set(path)) == len(path), "path revisits a cell"
    for index in range(len(path) - 1):
        r0, c0 = path[index]
        r1, c1 = path[index + 1]
        assert abs(r0 - r1) + abs(c0 - c1) == 1
        assert 0 <= r1 < rows and 0 <= c1 < cols
        assert grid[r1][c1] != "#"


def test_1_original_warehouse():
    grid, rows, cols, start, goal = load("warehouse_map.txt")
    astar_path, astar_expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    bfs_path, bfs_expanded = bfs(grid, rows, cols, start, goal)
    verify_path_is_valid(grid, rows, cols, start, goal, astar_path)
    verify_path_is_valid(grid, rows, cols, start, goal, bfs_path)
    assert path_length(astar_path) == path_length(bfs_path) == 40
    assert cost == 40
    assert astar_expanded == bfs_expanded == 64
    print(f"Test 1 PASSED: both find 40 moves and evaluate {astar_expanded} states")


def test_2_trivial():
    grid, rows, cols, start, goal = load("map_trivial.txt")
    path, expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    verify_path_is_valid(grid, rows, cols, start, goal, path)
    assert path_length(path) == cost == 1
    assert expanded == 2
    print(f"Test 2 PASSED: one-move solution; {expanded} states evaluated")


def test_3_no_solution():
    grid, rows, cols, start, goal = load("map_nosolution.txt")
    path, expanded, cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    assert path is None and cost is None
    assert expanded > 0
    print(f"Test 3 PASSED: no path; {expanded} states evaluated")


def test_4_alternative_paths_are_shortest():
    grid, rows, cols, start, goal = load("map_altpaths.txt")
    astar_path, astar_expanded, astar_cost = astar(grid, rows, cols, start, goal, heuristic=manhattan)
    bfs_path, bfs_expanded = bfs(grid, rows, cols, start, goal)
    verify_path_is_valid(grid, rows, cols, start, goal, astar_path)
    verify_path_is_valid(grid, rows, cols, start, goal, bfs_path)
    assert path_length(astar_path) == path_length(bfs_path) == 7
    assert astar_cost == 7
    assert astar_expanded <= bfs_expanded
    print(
        f"Test 4 PASSED: both find 7 moves; A* evaluated {astar_expanded}, "
        f"BFS evaluated {bfs_expanded} states"
    )


def test_5_parser_requires_exactly_one_start_and_goal():
    invalid_maps = [
        "###\n#S#\n###\n",
        "#####\n#SGG#\n#####\n",
        "#####\n#SSG#\n#####\n",
        "#####\n#S.GS\n#####\n",
    ]
    for map_text in invalid_maps:
        try:
            parse_map(map_text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid map was accepted: {map_text!r}")
    print("Test 5 PASSED: parser rejects missing or duplicate S/G markers")


def test_6_requested_heuristic_experiments():
    grid, rows, cols, start, goal = load("warehouse_map.txt")
    results = heuristic_experiments(grid, rows, cols, start, goal)
    assert [item["heuristic"] for item in results] == [
        "Zero heuristic (h=0)",
        "Euclidean distance",
        "Manhattan distance",
        "Manhattan x 2",
        "Manhattan x 5",
    ]
    assert [item["path_length"] for item in results] == [40, 40, 40, 40, 40]
    assert [item["expanded_states"] for item in results] == [64, 64, 64, 67, 84]
    for item in results:
        assert item["path_cost"] == item["path_length"]
    print("Test 6 PASSED: all five heuristic variants ran on the original map")
    for item in results:
        print(
            f"  {item['heuristic']}: path={item['path_length']}, "
            f"expanded={item['expanded_states']}"
        )


def test_7_branching_map_exposes_weighted_heuristic_tradeoff():
    grid, rows, cols, start, goal = load("map_altpaths.txt")
    results = heuristic_experiments(grid, rows, cols, start, goal)
    assert [item["path_length"] for item in results] == [7, 7, 7, 7, 9]
    assert [item["expanded_states"] for item in results] == [18, 14, 14, 14, 11]
    for item in results:
        verify_path_is_valid(grid, rows, cols, start, goal, item["path"])
        assert item["path_cost"] == item["path_length"]
    assert results[3]["path_length"] == 7  # 2x happens to remain optimal here.
    assert results[4]["path_length"] == 9  # 5x returns a valid but suboptimal route.
    print(
        "Test 7 PASSED: branching map shows 5x Manhattan returns a valid but "
        "suboptimal 9-move path after 11 expansions"
    )


def test_8_expansion_counts_use_one_convention():
    grid, rows, cols, start, goal = load("warehouse_map.txt")
    bfs_path, bfs_expanded = bfs(grid, rows, cols, start, goal)
    astar_path, astar_expanded, _ = astar(
        grid, rows, cols, start, goal, heuristic=manhattan
    )
    assert path_length(bfs_path) == path_length(astar_path) == 40
    assert bfs_expanded == astar_expanded == 64

    grid, rows, cols, start, goal = load("map_trivial.txt")
    trivial_path, trivial_expanded, _ = astar(
        grid, rows, cols, start, goal, heuristic=manhattan
    )
    assert path_length(trivial_path) == 1
    assert trivial_expanded == 2  # start and terminal goal are both evaluated.
    print("Test 8 PASSED: BFS and A* both count the terminal goal as expanded")


def test_9_zero_and_euclidean_match_bfs_optimum():
    grid, rows, cols, start, goal = load("warehouse_map.txt")
    bfs_path, _ = bfs(grid, rows, cols, start, goal)
    for heuristic in (lambda a, b: 0.0, euclidean):
        path, _, _ = astar(grid, rows, cols, start, goal, heuristic=heuristic)
        assert path_length(path) == path_length(bfs_path)
    print("Test 9 PASSED: zero and Euclidean heuristics preserve BFS optimality")


def main():
    test_1_original_warehouse()
    test_2_trivial()
    test_3_no_solution()
    test_4_alternative_paths_are_shortest()
    test_5_parser_requires_exactly_one_start_and_goal()
    test_6_requested_heuristic_experiments()
    test_7_branching_map_exposes_weighted_heuristic_tradeoff()
    test_8_expansion_counts_use_one_convention()
    test_9_zero_and_euclidean_match_bfs_optimum()
    print("\\nAll independent search tests passed.")


if __name__ == "__main__":
    main()
