"""Independent tests for the warehouse and revised locked-door planners."""

from planner import (
    LOCKED_DOOR_GOAL,
    LOCKED_DOOR_INITIAL_STATE,
    bfs_plan,
    goal_satisfied,
    make_drop_actions,
    make_locked_door_actions,
    make_move_actions,
    make_pickup_actions,
    validate_plan,
)

INITIAL_STATE = {"at(robot,a)", "at(package,a)"}
GOAL = {"at(package,c)"}


def test_A_warehouse_solvable():
    actions = make_move_actions() + make_pickup_actions() + make_drop_actions()
    plan, trace = bfs_plan(INITIAL_STATE, actions, GOAL)
    assert plan is not None
    for index, action in enumerate(plan):
        assert action.is_applicable(trace[index])
        assert action.apply(trace[index]) == trace[index + 1]
    assert goal_satisfied(trace[-1], frozenset(GOAL))
    assert [str(action) for action in plan] == [
        "PickUp(package,a)", "Move(a,b)", "Move(b,c)", "Drop(package,c)"
    ]
    print(f"Test A PASSED: warehouse plan = {[str(a) for a in plan]}")


def test_B_warehouse_unsolvable_without_pickup():
    actions = make_move_actions() + make_drop_actions()
    plan, trace = bfs_plan(INITIAL_STATE, actions, GOAL)
    assert plan is None and trace is None
    print("Test B PASSED: warehouse domain is unsolvable without PickUp")


def test_C_goal_is_about_package_not_robot():
    robot_only_at_c = frozenset({"at(robot,c)", "at(package,a)"})
    both_at_c = frozenset({"at(robot,c)", "at(package,c)"})
    assert not goal_satisfied(robot_only_at_c, frozenset(GOAL))
    assert goal_satisfied(both_at_c, frozenset(GOAL))
    print("Test C PASSED: package goal is checked independently")


def test_D_locked_door_shortest_plan_has_four_actions():
    actions = make_locked_door_actions()
    plan, trace = bfs_plan(LOCKED_DOOR_INITIAL_STATE, actions, LOCKED_DOOR_GOAL)
    assert plan is not None
    assert [action.name for action in plan] == [
        "PickKey", "Move(A,B)", "Unlock", "Move(B,C)"
    ]
    assert len(plan) == 4
    for index, action in enumerate(plan):
        assert action.is_applicable(trace[index])
        assert action.apply(trace[index]) == trace[index + 1]
    assert goal_satisfied(trace[-1], LOCKED_DOOR_GOAL)
    print(f"Test D PASSED: shortest locked-door plan = {[a.name for a in plan]}")


def test_E_locked_door_unsolvable_without_pick_key():
    actions = [action for action in make_locked_door_actions() if action.name != "PickKey"]
    plan, trace = bfs_plan(LOCKED_DOOR_INITIAL_STATE, actions, LOCKED_DOOR_GOAL)
    assert plan is None and trace is None
    print("Test E PASSED: no plan exists when PickKey is removed")


def test_F_invalid_llm_plan_reports_first_unsatisfied_precondition():
    proposed = ["Move(A,B)", "Unlock", "Move(B,C)"]
    result = validate_plan(LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), proposed)
    assert result["valid"] is False
    assert result["step"] == 2
    assert result["action"] == "Unlock"
    assert result["missing_positive_preconditions"] == ["has(key)"]
    assert result["violated_negative_preconditions"] == []
    print("Test F PASSED: first invalid action is Unlock; missing precondition has(key)")


def test_G_unknown_llm_action_is_reported():
    result = validate_plan(
        LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), ["FlyToMoon"]
    )
    assert result["valid"] is False
    assert result["step"] == 1
    assert result["reason"] == "unknown action"
    print("Test G PASSED: unknown proposed actions are rejected explicitly")


def test_H_bfs_respects_max_nodes_budget():
    try:
        bfs_plan(LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), LOCKED_DOOR_GOAL, max_nodes=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_nodes=0 must be rejected")
    limited_plan, _ = bfs_plan(
        LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), LOCKED_DOOR_GOAL, max_nodes=1
    )
    assert limited_plan is None
    print("Test H PASSED: BFS node budget is enforced")


def main():
    test_A_warehouse_solvable()
    test_B_warehouse_unsolvable_without_pickup()
    test_C_goal_is_about_package_not_robot()
    test_D_locked_door_shortest_plan_has_four_actions()
    test_E_locked_door_unsolvable_without_pick_key()
    test_F_invalid_llm_plan_reports_first_unsatisfied_precondition()
    test_G_unknown_llm_action_is_reported()
    test_H_bfs_respects_max_nodes_budget()
    print("\nAll independent planner tests passed.")


if __name__ == "__main__":
    main()
