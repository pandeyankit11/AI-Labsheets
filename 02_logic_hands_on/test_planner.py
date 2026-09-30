"""
Independent tests for planner.py (Task 3 of the lab).

These tests do NOT trust the LLM's explanation of its own code; they
re-run the generated planner and check the actual states/plans it
produces against the warehouse specification worked out by hand in
Task 0 / Task 1.
"""

from planner import (
    make_move_actions, make_pickup_actions, make_drop_actions,
    bfs_plan, goal_satisfied,
)

INITIAL_STATE = {"at(robot,a)", "at(package,a)"}
GOAL = {"at(package,c)"}


def test_A_solvable():
    """Test A: the original problem must be solvable, and every action
    in the returned plan must actually be applicable in the state
    before it (independently re-checked here, not just trusted)."""
    actions = make_move_actions() + make_pickup_actions() + make_drop_actions()
    plan, trace = bfs_plan(INITIAL_STATE, actions, GOAL)

    assert plan is not None, "Test A FAILED: expected a plan, got None"
    # Re-verify every step by hand: preconditions hold before, effects hold after
    for i, action in enumerate(plan):
        state_before = trace[i]
        state_after = trace[i + 1]
        assert action.is_applicable(state_before), (
            f"Test A FAILED: {action} not applicable in {sorted(state_before)}"
        )
        assert action.apply(state_before) == state_after, (
            f"Test A FAILED: {action} effects mismatch"
        )
    assert goal_satisfied(trace[-1], frozenset(GOAL)), "Test A FAILED: final state does not satisfy goal"
    print(f"Test A PASSED: plan = {[str(a) for a in plan]}")


def test_B_unsolvable_without_pickup():
    """Test B: with PickUp removed, no plan should exist -- the planner
    must report failure, not invent a way to move the package."""
    actions = make_move_actions() + make_drop_actions()  # no PickUp
    plan, trace = bfs_plan(INITIAL_STATE, actions, GOAL)
    assert plan is None, "Test B FAILED: planner found a plan without PickUp available!"
    print("Test B PASSED: planner correctly reports No plan found")


def test_C_goal_is_about_the_package_not_the_robot():
    """Test C: reaching C with the robot alone must NOT satisfy a goal
    that is stated in terms of the package's location."""
    robot_only_at_c = frozenset({"at(robot,c)", "at(package,a)"})
    both_at_c = frozenset({"at(robot,c)", "at(package,c)"})
    goal = frozenset(GOAL)

    assert not goal_satisfied(robot_only_at_c, goal), (
        "Test C FAILED: goal incorrectly satisfied by robot location alone"
    )
    assert goal_satisfied(both_at_c, goal), (
        "Test C FAILED: goal not recognised even when package is at C"
    )
    print("Test C PASSED: goal check correctly distinguishes robot vs. package location")


if __name__ == "__main__":
    test_A_solvable()
    test_B_unsolvable_without_pickup()
    test_C_goal_is_about_the_package_not_the_robot()
    print("\nAll independent tests passed.")
