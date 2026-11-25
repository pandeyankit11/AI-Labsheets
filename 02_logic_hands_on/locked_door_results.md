# Revised Locked-Door Planning: Task-by-Task Evidence

The original submission report, `AI_Lab_Logical_Planning_Submission.docx`, already covers the original warehouse sheet: initial state and goal, action preconditions/effects, the initial applicability question, the corrected manual warehouse plan and full state trace, the LLM prompt and planner mapping, original Tests A/B/C, the Logic-versus-Search explanation, reflection answers, and optional Prolog Tasks 6–8.

The corrected original warehouse plan is:

`PickUp(Package,A) -> Move(A,B) -> Move(B,C) -> Drop(Package,C)`

The handout's illustrative order `Move(A,B) -> PickUp(Package,B)` is invalid because moving the robot does not move the package.

This supplement records the additional requirements in `planning_lab_revised.pdf`, Tasks 6–7. The implementation uses the sheet's propositions and action preconditions/effects exactly.

## Task 6: Locked-door domain specification

Initial state: `I = {At(A), KeyAt(A)}`.

Goal: `G = {At(C)}`.

Exactly one of `At(A)`, `At(B)`, and `At(C)` must hold in every reachable state. Propositions absent from a state are false. The locked door is represented by the absence of `Unlocked`; the revised sheet's state vocabulary does not include a separate `Locked` proposition. The key remains held after pickup. There is no drop-key or relock action. All actions have unit cost.

| Action | Positive preconditions | Add effects | Delete effects |
|---|---|---|---|
| `Move(A,B)` | `At(A)` | `At(B)` | `At(A)` |
| `Move(B,A)` | `At(B)` | `At(A)` | `At(B)` |
| `PickKey` | `At(A)`, `KeyAt(A)` | `HasKey` | `KeyAt(A)` |
| `Unlock` | `At(B)`, `HasKey` | `Unlocked` | none |
| `Move(B,C)` | `At(B)`, `Unlocked` | `At(C)` | `At(B)` |
| `Move(C,B)` | `At(C)`, `Unlocked` | `At(B)` | `At(C)` |

Initially applicable actions are `Move(A,B)` and `PickKey`. `Unlock` is not initially applicable because `At(B)` and `HasKey` are absent. `Move(B,C)` is not initially applicable because `At(B)` and `Unlocked` are absent.

## Task 6(a)–(b): Shortest plan and complete state trace

BFS returns the shortest four-action plan:

`PickKey -> Move(A,B) -> Unlock -> Move(B,C)`

| State | Facts after the action |
|---|---|
| `S0` | `{At(A), KeyAt(A)}` |
| `S1` after `PickKey` | `{At(A), HasKey}` |
| `S2` after `Move(A,B)` | `{At(B), HasKey}` |
| `S3` after `Unlock` | `{At(B), HasKey, Unlocked}` |
| `S4` after `Move(B,C)` | `{At(C), HasKey, Unlocked}` |

`S4` satisfies the goal because `At(C)` is present. The tests replay each action, verify its preconditions against the preceding state, verify the effects, check the goal, and assert that exactly one `At(·)` proposition holds in every trace state.

## Task 6(c): Remove `PickKey`

With `PickKey` removed, no plan exists. `HasKey` is initially false and no remaining action can add it. Since `Unlock` requires `HasKey`, the door can never become unlocked, so `Move(B,C)` cannot be applied. BFS returns `(None, None)`. This is checked by `test_F_locked_door_unsolvable_without_pick_key`.

## Task 6(d): Validate the proposed sequence

Proposed sequence: `Move(A,B) -> Unlock -> Move(B,C)`.

`Move(A,B)` is valid and produces `{At(B), KeyAt(A)}`. `Unlock` is the first invalid action: `At(B)` is true, but `HasKey` is false. Validation stops at step 2 and reports missing positive precondition `HasKey`. The final move is not attempted.

## Task 7: LLM plan comparison and repair

The candidate-generation prompt supplied the exact initial state, goal, and six-action table from the revised sheet, with this instruction: “Return only a sequence of the six permitted actions. Do not use any other action.” The prompt did not show the BFS solution.

**Initial candidate:** `Move(A,B) -> Unlock -> Move(B,C)`.

**Independent validator result:** invalid at step 2 (`Unlock`); missing `HasKey`. The candidate does not reach the goal.

**Feedback supplied to the generator:** “Your proposed `Unlock` action is not applicable: the current state does not contain `HasKey`. Revise the plan using only the six listed actions.”

**Revised candidate:** `PickKey -> Move(A,B) -> Unlock -> Move(B,C)`.

**Validator result:** valid; final state contains `At(C)`; plan length is 4, equal to the BFS shortest-plan length. `test_L_llm_plan_repair_matches_bfs_shortest_length` checks this exact outcome. This is the candidate/revision generated in this ChatGPT session; it is not attributed to a separately queried external model.

A plausible narrative does not establish that preconditions hold. The independent validator computes each action transition from the current fact set and catches the missing key that a plausible explanation can overlook.

## Optional Prolog extension

The original logic submission report already records the optional Prolog work for the original warehouse domain and reports execution with SWI-Prolog 9.0.4: `can_move(a,b)` succeeds, `can_move(a,c)` fails, `valid_move(a,b)` and `valid_move(b,c)` succeed, `valid_move(a,c)` fails, and `reduce_speed` succeeds by the chain `wet_road -> slippery -> reduce_speed`. The revised sheet labels its Prolog section optional; it is not required for revised Tasks 6–7.

## Reproduction

```bash
python3 test_planner.py
python3 planner.py
```

Captured outputs are saved in `planner_output.txt` and `test_results.txt`. The original DOCX report is retained unchanged. This Markdown file supplements it with the revised locked-door domain and LLM-plan comparison.
