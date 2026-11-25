# LLM Prompt Engineering and Iteration Record

## LLM used

ChatGPT was used as the software-development assistant for this lab. The model was used to help design and implement the Python agent and tests; the generated code was then executed and reviewed.

## Initial prompt

> Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem shown in the question sheet. Represent the warehouse as a two-dimensional grid, determine a collision-free path from `S` to `G`, avoid all obstacles, and print either the path or a suitable message if no path exists. Explain the selected search algorithm and why it is appropriate.

This is close to the prompt suggested by the lab sheet. It establishes the overall objective, but leaves details such as malformed-map behavior, coordinate convention, shortest-path optimality, and test coverage implicit.

## Refined prompt

> Implement a well-documented Python 3 goal-based agent for the exact 7-row warehouse map in `warehouse_map.txt`. Treat the grid as rectangular and use zero-based `(row, column)` coordinates internally. The only valid symbols are `#`, `.`, `S`, and `G`, and the map must contain exactly one start and one goal. The vehicle may move exactly one cell Up, Down, Left, or Right; reject moves outside the grid or into `#`. All moves have unit cost. Use a complete algorithm that returns a shortest path, not just a route that happens to work. Reconstruct and print both the coordinate path and named actions. Print a clear no-path message when the destination is unreachable. Validate malformed maps, missing/duplicate markers, unsupported symbols, and unequal row lengths. Include deterministic behavior, a route visualization, and automated tests for the supplied map, legal moves, obstacle avoidance, shortest-path length, a blocked/no-path map, and parser edge cases. Explain the agent's state, goal, actions, transition model, search strategy, and time/space complexity. Use only the Python standard library.

## Why the refined prompt is better

It turns a vague request into measurable acceptance criteria. It also specifies assumptions that otherwise tend to become hidden implementation choices:

- allowed symbols and unique `S`/`G` markers;
- rectangular input and zero-based coordinate representation;
- no diagonal moves and no movement through obstacles;
- unit action costs and a shortest-path requirement;
- behavior when the goal is unreachable;
- deterministic path selection and output requirements;
- parser validation and independent tests.

## Iteration and validation log

1. The first complete BFS draft executed on the supplied map and found a valid path.
2. Code review tightened the map parser: unsupported characters, including spaces embedded inside a map row, must be rejected rather than silently interpreted as traversable cells.
3. The first test run exposed an incorrect test assumption: it required the one selected route to contain every direction. A shortest route need not use all four actions. The test was corrected to check that all returned action names belong to the allowed set, and a separate test verifies that all four actions are available from an open central cell.
4. The suite was expanded to cover no-path and malformed-map cases, path continuity, obstacle avoidance, deterministic tie-breaking, route rendering, and shortest-path length.
5. Final validation completed with 20 passing unit tests and a successful `compileall` check.

The test correction is recorded because LLM-assisted engineering requires validating the tests themselves, not blindly accepting every generated assertion.

## Algorithm selected

Breadth-First Search was selected because the warehouse is a finite, four-connected grid and every action has cost one. BFS is complete and finds a minimum-move route when one exists. Its worst-case time and memory are `O(V + E)` and `O(V)` respectively; for this grid, `E = O(V)`, so both are `O(R × C)`.
