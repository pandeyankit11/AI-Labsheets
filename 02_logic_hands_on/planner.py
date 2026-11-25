"""Simple STRIPS-style planning agent using breadth-first search (BFS)."""

from collections import deque


class Action:
    """A STRIPS-style action with positive/negative preconditions and effects."""

    def __init__(self, name, pos_pre, neg_pre, pos_eff, neg_eff):
        self.name = name
        self.pos_pre = frozenset(pos_pre)
        self.neg_pre = frozenset(neg_pre)
        self.pos_eff = frozenset(pos_eff)
        self.neg_eff = frozenset(neg_eff)

    def is_applicable(self, state):
        return self.pos_pre.issubset(state) and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        return (state - self.neg_eff) | self.pos_eff

    def __repr__(self):
        return self.name


def goal_satisfied(state, goal):
    return goal.issubset(state)


def bfs_plan(initial_state, actions, goal, max_nodes=100000):
    """Return a shortest plan and state trace, or (None, None) if unreachable."""
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")

    visited = {initial_state}
    frontier = deque([(initial_state, [], [initial_state])])
    expanded = 0

    while frontier:
        if expanded >= max_nodes:
            break
        state, plan, trace = frontier.popleft()
        expanded += 1
        if goal_satisfied(state, goal):
            return plan, trace

        for action in actions:
            if action.is_applicable(state):
                new_state = action.apply(state)
                if new_state in visited:
                    continue
                visited.add(new_state)
                frontier.append((new_state, plan + [action], trace + [new_state]))

    return None, None


def validate_plan(initial_state, actions, proposed_plan):
    """Validate a proposed plan and report its first failed precondition.

    The proposed actions may be Action objects or their exact action names.
    On failure, the result includes the 1-based step, action, reason,
    missing positive preconditions, and violated negative preconditions.
    """
    actions_by_name = {action.name: action for action in actions}
    state = frozenset(initial_state)
    trace = [state]

    for index, proposed_action in enumerate(proposed_plan, start=1):
        action_name = proposed_action.name if isinstance(proposed_action, Action) else str(proposed_action)
        action = actions_by_name.get(action_name)
        if action is None:
            return {
                "valid": False,
                "step": index,
                "action": action_name,
                "reason": "unknown action",
                "missing_positive_preconditions": [],
                "violated_negative_preconditions": [],
                "trace": trace,
                "final_state": state,
            }

        missing_positive = sorted(action.pos_pre - state)
        violated_negative = sorted(action.neg_pre & state)
        if missing_positive or violated_negative:
            return {
                "valid": False,
                "step": index,
                "action": action.name,
                "reason": "preconditions not satisfied",
                "missing_positive_preconditions": missing_positive,
                "violated_negative_preconditions": violated_negative,
                "trace": trace,
                "final_state": state,
            }

        state = action.apply(state)
        trace.append(state)

    return {"valid": True, "trace": trace, "final_state": state}


# Original warehouse domain
LOCATIONS = ["a", "b", "c"]
CONNECTIONS = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]


def make_move_actions():
    return [
        Action(
            name=f"Move({x},{y})",
            pos_pre={f"at(robot,{x})"},
            neg_pre=set(),
            pos_eff={f"at(robot,{y})"},
            neg_eff={f"at(robot,{x})"},
        )
        for x, y in CONNECTIONS
    ]


def make_pickup_actions(locations=LOCATIONS):
    return [
        Action(
            name=f"PickUp(package,{location})",
            pos_pre={f"at(robot,{location})", f"at(package,{location})"},
            neg_pre=set(),
            pos_eff={"holding(package)"},
            neg_eff={f"at(package,{location})"},
        )
        for location in locations
    ]


def make_drop_actions(locations=LOCATIONS):
    return [
        Action(
            name=f"Drop(package,{location})",
            pos_pre={f"at(robot,{location})", "holding(package)"},
            neg_pre=set(),
            pos_eff={f"at(package,{location})"},
            neg_eff={"holding(package)"},
        )
        for location in locations
    ]


# ---------------------------------------------------------------------------
# Revised locked-door domain (exact propositions from planning_lab_revised.pdf)
# ---------------------------------------------------------------------------

LOCKED_DOOR_INITIAL_STATE = {"At(A)", "KeyAt(A)"}
LOCKED_DOOR_GOAL = {"At(C)"}
LOCKED_DOOR_LOCATIONS = {"At(A)", "At(B)", "At(C)"}


def make_locked_door_actions():
    """Encode the six actions exactly as listed in the revised question sheet.

    The absence of Unlocked means the passage is still locked. There is no
    separate Locked proposition in the sheet's declared state vocabulary.
    """
    return [
        Action("Move(A,B)", {"At(A)"}, set(), {"At(B)"}, {"At(A)"}),
        Action("Move(B,A)", {"At(B)"}, set(), {"At(A)"}, {"At(B)"}),
        Action("PickKey", {"At(A)", "KeyAt(A)"}, set(), {"HasKey"}, {"KeyAt(A)"}),
        Action("Unlock", {"At(B)", "HasKey"}, set(), {"Unlocked"}, set()),
        Action("Move(B,C)", {"At(B)", "Unlocked"}, set(), {"At(C)"}, {"At(B)"}),
        Action("Move(C,B)", {"At(C)", "Unlocked"}, set(), {"At(B)"}, {"At(C)"}),
    ]


def exactly_one_robot_location(state):
    """Return True iff exactly one of At(A), At(B), At(C) is present."""
    return len(LOCKED_DOOR_LOCATIONS.intersection(state)) == 1


def print_result(label, plan, trace):
    print(f"--- {label} ---")
    if plan is None:
        print("No plan found\n")
        return
    print(f"Plan length: {len(plan)}")
    print(f"S0: {sorted(trace[0])}")
    for index, action in enumerate(plan):
        print(f"  --{action}-->")
        print(f"S{index + 1}: {sorted(trace[index + 1])}")
    print()


def main():
    # Original warehouse domain.
    initial_state = {"at(robot,a)", "at(package,a)"}
    goal = {"at(package,c)"}
    actions_full = make_move_actions() + make_pickup_actions() + make_drop_actions()
    plan_a, trace_a = bfs_plan(initial_state, actions_full, goal)
    print_result("Warehouse test (solvable)", plan_a, trace_a)

    actions_no_pickup = make_move_actions() + make_drop_actions()
    plan_b, trace_b = bfs_plan(initial_state, actions_no_pickup, goal)
    print_result("Warehouse test (PickUp removed)", plan_b, trace_b)

    # Revised locked-door domain.
    print("--- Revised locked-door domain ---")
    locked_actions = make_locked_door_actions()
    plan, trace = bfs_plan(LOCKED_DOOR_INITIAL_STATE, locked_actions, LOCKED_DOOR_GOAL)
    print_result("Locked-door shortest plan", plan, trace)
    print(
        "Exactly one At(·) location in each trace state:",
        all(exactly_one_robot_location(state) for state in trace),
    )

    no_key_actions = [action for action in locked_actions if action.name != "PickKey"]
    no_key_plan, _ = bfs_plan(LOCKED_DOOR_INITIAL_STATE, no_key_actions, LOCKED_DOOR_GOAL)
    print(f"Plan without PickKey: {'found' if no_key_plan else 'No plan found'}")

    # Task 6(d): first error in the proposed plan.
    invalid_plan = ["Move(A,B)", "Unlock", "Move(B,C)"]
    validation = validate_plan(LOCKED_DOOR_INITIAL_STATE, locked_actions, invalid_plan)
    print(
        "Invalid LLM plan validation: "
        f"valid={validation['valid']}, first_failure_step={validation['step']}, "
        f"action={validation['action']}, "
        f"missing={validation['missing_positive_preconditions']}"
    )

    # Task 7: validator feedback is applied to a revised candidate.
    revised_llm_plan = ["PickKey", "Move(A,B)", "Unlock", "Move(B,C)"]
    revised_result = validate_plan(
        LOCKED_DOOR_INITIAL_STATE, locked_actions, revised_llm_plan
    )
    print(
        "Revised LLM plan: "
        f"valid={revised_result['valid']}, "
        f"reaches_goal={goal_satisfied(revised_result['final_state'], LOCKED_DOOR_GOAL)}, "
        f"length={len(revised_llm_plan)}, "
        f"matches_BFS_length={plan is not None and len(revised_llm_plan) == len(plan)}"
    )


if __name__ == "__main__":
    main()
