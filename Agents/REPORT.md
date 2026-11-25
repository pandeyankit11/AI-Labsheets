# Agents Laboratory Exercise
## Constructing a Goal-Based Agent using a Large Language Model

**Problem:** Warehouse navigation  
**Implementation:** Python 3, standard library only  
**LLM used during development:** ChatGPT  
**Search strategy:** Breadth-First Search (BFS)

---

## Task 1 – Understanding the Problem

### 1. What is the environment?

The environment is a **known, static, discrete two-dimensional warehouse grid**. Each grid cell is either an obstacle (`#`) or traversable space (`.`); the unique start (`S`) and goal (`G`) cells are also traversable. The supplied map has 7 rows and 21 columns. The outer border is made of obstacles. The agent is assumed to know the complete map before planning; no moving obstacles or other vehicles are included in this exercise.

### 2. What is the goal of the agent?

The agent must move the vehicle from `S` to `G` without entering an obstacle or leaving the grid. Since each move has the same cost of one step, the natural objective is to find a **shortest collision-free path**, not merely any path that eventually reaches the goal.

### 3. What actions are available?

The action set is:

- `Up`: change the row by −1.
- `Down`: change the row by +1.
- `Left`: change the column by −1.
- `Right`: change the column by +1.

An action is legal only when the destination cell is inside the grid and is not `#`. Each action changes the position by exactly one cell; diagonal movement is not allowed.

### 4. What information must the agent maintain?

At minimum, the agent needs:

1. **Environment/map:** the grid and which cells are blocked.
2. **Current state:** the vehicle's current `(row, column)` position.
3. **Goal state:** the destination coordinate containing `G`.
4. **Action/transition model:** the four possible moves and the rule for checking whether a move is legal.
5. **Planning memory:** the BFS frontier, the visited set, and a parent pointer for each discovered state so that the path can be reconstructed.

The implementation represents the current search state as a grid coordinate. During planning, the `parent` dictionary records how each discovered coordinate was reached. After the plan is returned, the path is converted into named movement actions.

### 5. Why is this a goal-based agent rather than a simple reflex agent?

A simple reflex agent selects an action from the current percept using a fixed condition/action rule. That is insufficient here: choosing a direction based only on the vehicle's immediate position can lead into a dead end or cause it to repeatedly move without reaching `G`.

This agent has an explicit goal coordinate and evaluates future reachable states. It searches through alternative routes, avoids obstacles, and selects a sequence of actions whose final state is the goal. Its decisions therefore depend on the goal and a plan, not only on the current cell.

### Think About It – What if the warehouse becomes twice as large?

BFS remains correct and is still a sensible baseline for a moderately larger **unweighted** grid. Its worst-case time and memory are both `O(V + E)`, where `V` is the number of traversable cells and `E` is the number of legal neighbor connections. In a four-connected grid, `E` is at most about `4V`, so this is `O(R × C)` in the number of rows `R` and columns `C`.

The phrase “twice as large” can mean different things. If the number of traversable cells doubles, the worst-case work and stored search data can roughly double. If both grid dimensions double, the number of cells becomes roughly four times larger. A much larger map can therefore make BFS expensive in memory and search time, especially when the goal is far away or many cells are reachable.

For larger maps, A* with an admissible Manhattan-distance heuristic is a useful alternative for four-direction, unit-cost movement. It can focus exploration toward the goal while preserving shortest-path optimality when the heuristic is admissible and the implementation handles the frontier correctly. Other options include bidirectional BFS, hierarchical path planning, and Dijkstra's algorithm if movement costs differ. Dynamic obstacles, incomplete maps, multiple vehicles, and sensor uncertainty would require additional state, collision handling, and possibly replanning while the vehicle moves.

---

## Task 2 – Designing the Agent

### Design specification

| Component | Design used in this lab |
|---|---|
| Environment | Static rectangular grid; `#` is blocked and `.`, `S`, `G` are traversable. |
| Current state | Vehicle coordinate `(row, column)`, represented with zero-based indices. |
| Initial state | Coordinate of the unique `S` cell. |
| Goal | Coordinate of the unique `G` cell. |
| Actions | Up, Down, Left, Right; one orthogonal cell per action. |
| Transition model | Apply the action offset only if the resulting cell is in bounds and not an obstacle. |
| Goal test | Current coordinate equals the goal coordinate. |
| Decision-making | BFS explores legal states in increasing path-length order, stores parent links, and reconstructs the shortest path. |
| Output | Ordered list of coordinates, named movement actions, and a map with the route marked. |

### Block diagram

The architecture is shown in [`agent_architecture.svg`](agent_architecture.svg). The environment supplies the map and current position. The state representation and goal are passed to the BFS decision component. The selected action changes the vehicle's position in the environment; the updated position becomes the next state. The loop continues until the goal test succeeds or the planner establishes that no path exists.

### Why BFS?

The state space is finite, movement is deterministic, and every legal action has equal cost (one move). BFS uses a FIFO queue, so it examines states in nondecreasing path length. It is therefore:

- **Complete:** if a path exists in this finite grid, BFS will eventually find it.
- **Optimal for this problem:** the first discovered route to the goal has the fewest moves because every edge has unit cost.
- **Easy to verify:** each transition is one of four legal orthogonal moves, and parent links reconstruct the chosen route.

For `V` traversable cells and `E` legal connections, BFS takes `O(V + E)` time and `O(V)` auxiliary space. On a four-neighbor rectangular grid this is `O(R × C)` time and space in the worst case.

---

## Task 3 – Prompt Engineering and Testing

### Initial prompt

The initial prompt followed the assignment's suggested specification: implement a documented Python program for the exact warehouse map, avoid obstacles, find a path from `S` to `G`, print a route or a no-path message, and explain the chosen search algorithm.

### Refined prompt

The prompt was made more precise to reduce ambiguity and improve testability:

> Implement a well-documented Python 3 goal-based agent for the exact 7-row warehouse map in `warehouse_map.txt`. Treat the grid as rectangular and use zero-based `(row, column)` coordinates internally. The only valid symbols are `#`, `.`, `S`, and `G`, and the map must contain exactly one start and one goal. The vehicle may move exactly one cell Up, Down, Left, or Right; reject moves outside the grid or into `#`. All moves have unit cost. Use a complete algorithm that returns a shortest path, not just a route that happens to work. Reconstruct and print both the coordinate path and named actions. Print a clear no-path message when the destination is unreachable. Validate malformed maps, missing/duplicate markers, unsupported symbols, and unequal row lengths. Include deterministic behavior, a route visualization, and automated tests for the supplied map, legal moves, obstacle avoidance, shortest-path length, a blocked/no-path map, and parser edge cases. Explain the agent's state, goal, actions, transition model, search strategy, and time/space complexity. Use only the Python standard library.

### 1. Did the LLM generate a working program on the first attempt?

The first complete BFS draft ran on the supplied map and found a valid route. However, it was not treated as a finished submission after that single run. Review identified that input validation and the acceptance tests needed to be stricter. The symbol whitelist was tightened so that a space or any other unsupported character could not silently be treated as free space. The tests were also corrected to check that every produced action is legal, rather than incorrectly requiring the selected route to use all four possible directions. The route itself remained a valid 20-move BFS solution.

This is an important distinction: a program producing one plausible output is not sufficient evidence that it handles malformed input or edge cases correctly. The refined prompt and repeated tests improved confidence in the final version.

### 2. If not, how can the prompt be improved?

A prompt should specify observable acceptance criteria rather than only saying “write a program.” In this exercise, the refined prompt explicitly states the map dimensions and symbols, coordinate convention, exact action model, unit-cost objective, shortest-path requirement, error cases, output format, and tests. These details reduce the chance that the LLM will silently make assumptions, use diagonal movement, treat an obstacle as traversable, return a non-shortest route, or omit the no-path case.

The prompt should also ask for code that can be inspected and tested, not merely code accompanied by a confident explanation.

### 3. What search algorithm did the LLM choose?

The implementation uses **Breadth-First Search (BFS)**. It maintains a FIFO queue of frontier positions, a visited set to avoid revisiting cells, and a parent dictionary to reconstruct the route when the goal is reached.

### 4. Why did the LLM select this algorithm?

BFS is the simplest appropriate optimal search strategy for this particular problem: the grid is finite, every move has identical cost, and the required result is a minimum-move path. DFS would not guarantee the shortest route. Dijkstra's algorithm would also be optimal but is unnecessary when every edge cost is one. A* could reduce exploration on larger maps, but requires a heuristic and is not necessary for this small map.

---

## Execution and Verification Results

### Supplied warehouse

The final planner finds a path of **20 moves** from `(1, 1)` to `(1, 19)`. The action sequence is:

`Right → Right → Right → Down → Right → Right → Right → Up → Right → Right → Right → Right → Right → Right → Right → Right → Right → Right → Right → Right`

The path enters only free cells, uses only orthogonal one-cell moves, and reaches `G`. An independent distance-only BFS in the test suite confirms that 20 is the minimum number of moves on the supplied map.

The program prints a route map using `*` for intermediate path cells. The exact captured output is in `lab_output.txt`.

### Automated tests

Run:

```bash
python3 -m unittest -v test_goal_based_agent.py
```

**Result: 20 tests passed, 0 failed.** Tests cover:

- parsing the supplied map and verifying dimensions/start/goal;
- empty, ragged, invalid-symbol, missing-marker, and duplicate-marker maps;
- out-of-bounds checks and the exact four legal movement directions;
- finding a path, obstacle avoidance, orthogonal movement, action conversion, and deterministic tie-breaking;
- shortest-path length on the supplied map;
- a direct corridor, an unreachable goal, route rendering, and rejection of non-adjacent path steps.

The final test output is preserved in `test_results.txt`. `python3 -m compileall -q goal_based_agent.py test_goal_based_agent.py` also completes successfully.

---

## Critical Evaluation: Strengths and Limitations of LLM-Assisted Development

**Strengths:** An LLM can quickly draft a readable program, suggest a suitable search algorithm, explain its data structures, and propose tests. It is especially useful for iterating on a precise specification and producing documentation alongside code.

**Limitations and risks:** Generated code can contain incorrect assumptions or incomplete validation. A plausible route on one map does not prove correctness. Tests themselves can also be wrong: the first test draft mistakenly expected the chosen path to contain all four movement directions. That assertion was corrected after checking what the test was actually measuring. The final code therefore needed manual inspection, execution, edge-case testing, and an independent shortest-distance comparison.

The implementation also models a static, fully known, single-vehicle warehouse. It does not handle moving obstacles, traffic from other vehicles, sensor errors, uncertain maps, turn costs, battery constraints, or changes in the environment after planning. BFS can become memory-intensive on very large maps. A production warehouse system would need a richer environment model, continuous state updates, safety constraints, and replanning.

**Conclusion:** LLM assistance accelerated implementation and explanation, but it did not replace problem specification, code review, or independent testing. The final result is supported by executable tests and captured outputs rather than by the LLM's claim that the code is correct.
