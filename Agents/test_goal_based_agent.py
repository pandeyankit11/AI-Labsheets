"""Independent unit tests for the goal-based warehouse navigation lab."""

import unittest

from goal_based_agent import (
    DEFAULT_MAP,
    GoalBasedWarehouseAgent,
    Warehouse,
    path_to_actions,
    render_path,
)


class WarehouseParsingTests(unittest.TestCase):
    def test_supplied_map_dimensions_and_markers(self):
        warehouse = Warehouse.from_text(DEFAULT_MAP)
        self.assertEqual((warehouse.rows, warehouse.columns), (7, 21))
        self.assertEqual(warehouse.start, (1, 1))
        self.assertEqual(warehouse.goal, (1, 19))

    def test_rejects_empty_map(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Warehouse.from_text("\n  \n")

    def test_rejects_missing_start(self):
        with self.assertRaisesRegex(ValueError, "exactly one S"):
            Warehouse.from_text("###\n#G#\n###")

    def test_rejects_missing_goal(self):
        with self.assertRaisesRegex(ValueError, "exactly one G"):
            Warehouse.from_text("###\n#S#\n###")

    def test_rejects_duplicate_start_and_goal(self):
        with self.assertRaisesRegex(ValueError, "exactly one S"):
            Warehouse.from_text("S.GS")
        with self.assertRaisesRegex(ValueError, "exactly one G"):
            Warehouse.from_text("SGG")

    def test_rejects_unsupported_symbol(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            Warehouse.from_text("SxG")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            Warehouse.from_text("S G")

    def test_rejects_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            Warehouse.from_text("S..\n..G.")

    def test_out_of_bounds_is_not_free(self):
        warehouse = Warehouse.from_text("#####\n#S.G#\n#####")
        self.assertFalse(warehouse.is_free((-1, 1)))
        self.assertFalse(warehouse.is_free((1, 5)))


class GoalBasedAgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = Warehouse.from_text(DEFAULT_MAP)
        self.agent = GoalBasedWarehouseAgent(self.environment)

    def test_finds_path_on_supplied_warehouse(self):
        path = self.agent.plan()
        self.assertIsNotNone(path)
        self.assertEqual(path[0], self.environment.start)
        self.assertEqual(path[-1], self.environment.goal)

    def test_available_actions_are_exactly_four_orthogonal_moves(self):
        open_grid = Warehouse.from_text("#####\n#...#\n#.S.#\n#..G#\n#####")
        actions = [action for action, _position in open_grid.successors((2, 2))]
        self.assertEqual(actions, ["Up", "Down", "Left", "Right"])

    def test_every_path_cell_is_free(self):
        path = self.agent.plan()
        self.assertTrue(all(self.environment.is_free(position) for position in path))

    def test_every_path_step_is_one_orthogonal_move(self):
        path = self.agent.plan()
        for current, nxt in zip(path, path[1:]):
            distance = abs(current[0] - nxt[0]) + abs(current[1] - nxt[1])
            self.assertEqual(distance, 1)

    def test_actions_match_path_length(self):
        path = self.agent.plan()
        actions = path_to_actions(path)
        self.assertEqual(len(actions), len(path) - 1)
        self.assertTrue(set(actions).issubset({"Up", "Down", "Left", "Right"}))

    def test_path_is_shortest_for_supplied_map(self):
        # Independent distance-only BFS provides an oracle for optimal length.
        start, goal = self.environment.start, self.environment.goal
        queue = [(start, 0)]
        seen = {start}
        expected_distance = None
        for position, distance in queue:
            if position == goal:
                expected_distance = distance
                break
            for _action, nxt in self.environment.successors(position):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        self.assertIsNotNone(expected_distance)
        self.assertEqual(len(self.agent.plan()) - 1, expected_distance)
        self.assertEqual(expected_distance, 20)

    def test_no_path_is_reported(self):
        blocked = Warehouse.from_text("#####\n#S#G#\n#####")
        self.assertIsNone(GoalBasedWarehouseAgent(blocked).plan())

    def test_open_corridor_returns_direct_route(self):
        corridor = Warehouse.from_text("#####\n#S.G#\n#####")
        path = GoalBasedWarehouseAgent(corridor).plan()
        self.assertEqual(path, [(1, 1), (1, 2), (1, 3)])
        self.assertEqual(path_to_actions(path), ["Right", "Right"])

    def test_deterministic_tie_breaking(self):
        open_grid = Warehouse.from_text("#####\n#S..#\n#...#\n#..G#\n#####")
        first = GoalBasedWarehouseAgent(open_grid).plan()
        second = GoalBasedWarehouseAgent(open_grid).plan()
        self.assertEqual(first, second)

    def test_renderer_marks_only_intermediate_route_cells(self):
        corridor = Warehouse.from_text("#####\n#S.G#\n#####")
        path = GoalBasedWarehouseAgent(corridor).plan()
        rendered = render_path(corridor, path)
        self.assertIn("#S*G#", rendered)
        self.assertEqual(rendered.count("*"), 1)

    def test_none_path_has_no_route_markers(self):
        corridor = Warehouse.from_text("#####\n#S#G#\n#####")
        self.assertNotIn("*", render_path(corridor, None))

    def test_path_to_actions_rejects_non_adjacent_steps(self):
        with self.assertRaisesRegex(ValueError, "non-adjacent"):
            path_to_actions([(0, 0), (2, 2)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
