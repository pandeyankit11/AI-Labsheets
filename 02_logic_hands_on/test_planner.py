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
    exactly_one_robot_location,
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


def test_C_initial_warehouse_applicability():
    actions = make_move_actions() + make_pickup_actions() + make_drop_actions()
    by_name = {action.name: action for action in actions}
    initial = frozenset(INITIAL_STATE)
    assert by_name["PickUp(package,a)"].is_applicable(initial)
    assert by_name["Move(a,b)"].is_applicable(initial)
    assert not by_name["Drop(package,c)"].is_applicable(initial)
    print("Test C PASSED: PickUp at A and Move(A,B) are applicable; Drop at C is not")


def test_D_goal_is_about_package_not_robot():
    robot_only_at_c = frozenset({"at(robot,c)", "at(package,a)"})
    both_at_c = frozenset({"at(robot,c)", "at(package,c)"})
    assert not goal_satisfied(robot_only_at_c, frozenset(GOAL))
    assert goal_satisfied(both_at_c, frozenset(GOAL))
    print("Test D PASSED: package goal is checked independently")


def test_E_locked_door_shortest_plan_has_four_actions():
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
    print(f"Test E PASSED: shortest locked-door plan = {[a.name for a in plan]}")


def test_F_locked_door_unsolvable_without_pick_key():
    actions = [action for action in make_locked_door_actions() if action.name != "PickKey"]
    plan, trace = bfs_plan(LOCKED_DOOR_INITIAL_STATE, actions, LOCKED_DOOR_GOAL)
    assert plan is None and trace is None
    print("Test F PASSED: no plan exists when PickKey is removed")


def test_G_invalid_llm_plan_reports_first_unsatisfied_precondition():
    proposed = ["Move(A,B)", "Unlock", "Move(B,C)"]
    result = validate_plan(LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), proposed)
    assert result["valid"] is False
    assert result["step"] == 2
    assert result["action"] == "Unlock"
    assert result["missing_positive_preconditions"] == ["HasKey"]
    assert result["violated_negative_preconditions"] == []
    print("Test G PASSED: first invalid action is Unlock; missing precondition HasKey")


def test_H_unknown_llm_action_is_reported():
    result = validate_plan(
        LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), ["FlyToMoon"]
    )
    assert result["valid"] is False
    assert result["step"] == 1
    assert result["reason"] == "unknown action"
    print("Test H PASSED: unknown proposed actions are rejected explicitly")


def test_I_bfs_respects_max_nodes_budget():
    try:
        bfs_plan(
            LOCKED_DOOR_INITIAL_STATE,
            make_locked_door_actions(),
            LOCKED_DOOR_GOAL,
            max_nodes=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("max_nodes=0 must be rejected")
    limited_plan, _ = bfs_plan(
        LOCKED_DOOR_INITIAL_STATE,
        make_locked_door_actions(),
        LOCKED_DOOR_GOAL,
        max_nodes=1,
    )
    assert limited_plan is None
    print("Test I PASSED: BFS node budget is enforced")


def test_J_locked_door_initial_applicability_matches_action_table():
    actions = make_locked_door_actions()
    applicable = [
        action.name for action in actions
        if action.is_applicable(LOCKED_DOOR_INITIAL_STATE)
    ]
    assert applicable == ["Move(A,B)", "PickKey"]
    print("Test J PASSED: initially applicable actions are Move(A,B) and PickKey")


def test_K_locked_door_preserves_exactly_one_location():
    plan, trace = bfs_plan(
        LOCKED_DOOR_INITIAL_STATE, make_locked_door_actions(), LOCKED_DOOR_GOAL
    )
    assert plan is not None
    assert all(exactly_one_robot_location(state) for state in trace)
    assert trace[-1] == frozenset({"At(C)", "HasKey", "Unlocked"})
    print("Test K PASSED: all reachable plan states have exactly one At(·) proposition")


def test_L_llm_plan_repair_matches_bfs_shortest_length():
    actions = make_locked_door_actions()
    shortest, _ = bfs_plan(LOCKED_DOOR_INITIAL_STATE, actions, LOCKED_DOOR_GOAL)
    proposed = ["PickKey", "Move(A,B)", "Unlock", "Move(B,C)"]
    result = validate_plan(LOCKED_DOOR_INITIAL_STATE, actions, proposed)
    assert result["valid"]
    assert goal_satisfied(result["final_state"], LOCKED_DOOR_GOAL)
    assert len(proposed) == len(shortest) == 4
    print("Test L PASSED: revised LLM plan is valid, reaches the goal, and matches BFS length")


def main():
    test_A_warehouse_solvable()
    test_B_warehouse_unsolvable_without_pickup()
    test_C_initial_warehouse_applicability()
    test_D_goal_is_about_package_not_robot()
    test_E_locked_door_shortest_plan_has_four_actions()
    test_F_locked_door_unsolvable_without_pick_key()
    test_G_invalid_llm_plan_reports_first_unsatisfied_precondition()
    test_H_unknown_llm_action_is_reported()
    test_I_bfs_respects_max_nodes_budget()
    test_J_locked_door_initial_applicability_matches_action_table()
    test_K_locked_door_preserves_exactly_one_location()
    test_L_llm_plan_repair_matches_bfs_shortest_length()
    print("\nAll independent planner tests passed.")


if __name__ == "__main__":
    main()
