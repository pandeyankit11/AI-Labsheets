# Revised Locked-Door Planning Results

## Domain

Initial state:
- `at(robot,A)`
- `at(key,A)`
- `door_locked`

Goal: `at(robot,C)`.

The six STRIPS actions are `PickKey`, `Move(A,B)`, `Move(B,A)`, `Unlock`, `Move(B,C)`, and `Move(C,B)`. `Unlock` requires the robot at B, `has(key)`, and `door_locked`. `Move(B,C)` requires the robot at B and `door_unlocked`.

## BFS result

Shortest plan: `PickKey -> Move(A,B) -> Unlock -> Move(B,C)`.

Plan length: **4 actions**. The tests replay every action against the trace and check that the final state satisfies the goal.

## Required negative cases

1. Removing `PickKey` makes the goal unreachable because the robot cannot establish `has(key)`, a required precondition for `Unlock`.
2. The proposed sequence `Move(A,B) -> Unlock -> Move(B,C)` is invalid at step 2, `Unlock`, because `has(key)` is absent. `validate_plan()` reports the first unsatisfied precondition and does not apply later actions.

## Reproduction

```bash
python3 test_planner.py
python3 planner.py
```

The revised task's optional Prolog extension is not included; the required planning and plan-validation cases are implemented and tested in Python.
