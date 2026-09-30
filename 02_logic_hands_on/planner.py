"""
Simple STRIPS-style planning agent using Breadth-First Search (BFS).

Generated with the assistance of an LLM (Claude), following the exact
specification given in the prompt recorded in the lab report. Lightly
reviewed and re-tested independently (see test_planner.py).
"""

from collections import deque


class Action:
    """A STRIPS-style action.

    name            -- human readable label, e.g. "Move(a,b)"
    pos_pre         -- set of propositions that must be TRUE for the action to apply
    neg_pre         -- set of propositions that must be FALSE for the action to apply
    pos_eff         -- set of propositions added to the state after the action
    neg_eff         -- set of propositions removed from the state after the action
    """

    def __init__(self, name, pos_pre, neg_pre, pos_eff, neg_eff):
        self.name = name
        self.pos_pre = frozenset(pos_pre)
        self.neg_pre = frozenset(neg_pre)
        self.pos_eff = frozenset(pos_eff)
        self.neg_eff = frozenset(neg_eff)

    def is_applicable(self, state):
        """An action is applicable iff all positive preconditions hold
        and none of the negative preconditions hold in `state`."""
        return self.pos_pre.issubset(state) and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        """Remove negative effects, then add positive effects."""
        return (state - self.neg_eff) | self.pos_eff

    def __repr__(self):
        return self.name


def goal_satisfied(state, goal):
    """The goal is satisfied when every proposition in `goal` is present in `state`."""
    return goal.issubset(state)


def bfs_plan(initial_state, actions, goal, max_nodes=100000):
    """Breadth-first search over the state space.

    Returns (plan, trace) where `plan` is a list of Action objects and
    `trace` is the list of states from the initial state through the
    goal state (inclusive). Returns (None, None) if no plan exists.
    """
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal_satisfied(initial_state, goal):
        return [], [initial_state]

    visited = {initial_state}
    frontier = deque([(initial_state, [], [initial_state])])
    expanded = 0

    while frontier:
        state, plan, trace = frontier.popleft()
        expanded += 1
        if expanded > max_nodes:
            break
        for action in actions:
            if action.is_applicable(state):
                new_state = action.apply(state)
                if new_state in visited:
                    continue
                new_plan = plan + [action]
                new_trace = trace + [new_state]
                if goal_satisfied(new_state, goal):
                    return new_plan, new_trace
                visited.add(new_state)
                frontier.append((new_state, new_plan, new_trace))

    return None, None


# ---------------------------------------------------------------------------
# Warehouse domain
# ---------------------------------------------------------------------------

LOCATIONS = ["a", "b", "c"]
CONNECTIONS = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]


def make_move_actions():
    actions = []
    for (x, y) in CONNECTIONS:
        actions.append(Action(
            name=f"Move({x},{y})",
            pos_pre={f"at(robot,{x})"},
            neg_pre=set(),
            pos_eff={f"at(robot,{y})"},
            neg_eff={f"at(robot,{x})"},
        ))
    return actions


def make_pickup_actions(locations=LOCATIONS):
    actions = []
    for l in locations:
        actions.append(Action(
            name=f"PickUp(package,{l})",
            pos_pre={f"at(robot,{l})", f"at(package,{l})"},
            neg_pre=set(),
            pos_eff={"holding(package)"},
            neg_eff={f"at(package,{l})"},
        ))
    return actions


def make_drop_actions(locations=LOCATIONS):
    actions = []
    for l in locations:
        actions.append(Action(
            name=f"Drop(package,{l})",
            pos_pre={f"at(robot,{l})", "holding(package)"},
            neg_pre=set(),
            pos_eff={f"at(package,{l})"},
            neg_eff={"holding(package)"},
        ))
    return actions


def print_result(label, plan, trace):
    print(f"--- {label} ---")
    if plan is None:
        print("No plan found")
        print()
        return
    print(f"Plan length: {len(plan)}")
    print(f"S0: {sorted(trace[0])}")
    for i, action in enumerate(plan):
        print(f"  --{action}-->")
        print(f"S{i+1}: {sorted(trace[i+1])}")
    print()


if __name__ == "__main__":
    initial_state = {"at(robot,a)", "at(package,a)"}
    goal = {"at(package,c)"}

    # ---- Test A: solvable problem (full action set) ----
    actions_full = make_move_actions() + make_pickup_actions() + make_drop_actions()
    plan_a, trace_a = bfs_plan(initial_state, actions_full, goal)
    print_result("Test A (solvable)", plan_a, trace_a)

    # ---- Test B: impossible problem (no PickUp action available) ----
    actions_no_pickup = make_move_actions() + make_drop_actions()
    plan_b, trace_b = bfs_plan(initial_state, actions_no_pickup, goal)
    print_result("Test B (PickUp removed -> should be unsolvable)", plan_b, trace_b)

    # ---- Test C: irrelevant robot-only movement must not fool the goal check ----
    print("--- Test C (goal-check sanity) ---")
    state_robot_at_c_only = frozenset({"at(robot,c)", "at(package,a)"})
    state_both_at_c = frozenset({"at(robot,c)", "at(package,c)"})
    test_goal = frozenset({"at(package,c)"})
    print(f"goal_satisfied(robot@C, package@A) = {goal_satisfied(state_robot_at_c_only, test_goal)}  (expected False)")
    print(f"goal_satisfied(robot@C, package@C) = {goal_satisfied(state_both_at_c, test_goal)}  (expected True)")
    print(f"Test A plan actually used: {[str(a) for a in plan_a]}")
    print()
