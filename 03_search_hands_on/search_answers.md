# Search and A* — Written Answers and Evaluation

Source: `AI_Lab_Search_AStar_Submission.docx` (Search handout, Tasks 0–9).

This file is the written companion to the implementation. The DOCX is the question sheet; it is not being treated as a completed answer report. Reproducible measurements are in `search_output.txt`, `test_results.txt`, and `heuristic_experiments.md`.

## 1. Problem formulation

The warehouse is represented as a four-connected grid.

| Component | Definition |
|---|---|
| State | The agent's grid coordinate `(row, column)`. |
| Initial state | The coordinate marked `S`. |
| Goal test | The current coordinate equals the coordinate marked `G`. |
| Actions | `Up`, `Down`, `Left`, and `Right`. |
| Transition model | Apply one action if the resulting coordinate is inside the grid and is not a wall (`#`). |
| Step cost | 1 for every legal move. |
| Path cost | Sum of the unit step costs, equal to the number of moves. |

The map parser pads shorter rows with walls and requires exactly one `S`, exactly one `G`, and only the symbols `#`, `.`, `S`, and `G`. Rejecting malformed maps is preferable to silently selecting the first start or goal and solving a different problem.

## 2. BFS and A* design

### Breadth-first search

BFS uses a FIFO queue. It marks a state visited when it is enqueued, records its predecessor, and stops when the goal is dequeued. Marking states at enqueue time prevents duplicate frontier entries. Since every move has the same cost, BFS returns a minimum-move path.

For a graph with (V) reachable states and (E) legal transitions, the usual time and space bounds are (O(V+E)) and (O(V)), respectively.

### A* search

A* orders frontier entries by

[
f(n)=g(n)+h(n),
]

where (g(n)) is the path cost from the start and (h(n)) estimates the remaining cost to the goal. For this four-connected grid with unit move costs, Manhattan distance is

[
h_M(n)=|r_n-r_G|+|c_n-c_G|.
]

Euclidean distance is

[
h_E(n)=sqrt{(r_n-r_G)^2+(c_n-c_G)^2}.
]

Both are lower bounds on the number of four-connected moves and are consistent for this movement model. With a consistent heuristic, graph-search A* returns an optimal path. The implementation also permits reopening a previously expanded state when a strictly cheaper (g)-score is discovered, which is important when experimenting with inconsistent or weighted heuristics.

The implementation reconstructs the final path from predecessor links. It reports path length, path cost, and expansion events separately.

## 3. Why the heuristic variants matter

The experiment compares five configurations:

1. (h(n)=0)
2. Euclidean distance
3. Manhattan distance
4. (2h_M(n))
5. (5h_M(n))

The zero heuristic reduces A* to uniform-cost search; with unit costs, its ordering behaves like BFS. Manhattan and Euclidean are admissible for the stated grid model. Multiplying Manhattan by 2 or 5 can make it overestimate the remaining cost, so those weighted variants do not retain the general optimality guarantee of ordinary A*.

A smaller expansion count is not, by itself, proof of a better search result. Path validity and path optimality must be checked separately.

## 4. Measured results

The expansion convention is consistent across the implementations: count each non-stale frontier removal that is evaluated, including the terminal goal. A* re-expansions after a strict improvement in (g) count again.

### Original warehouse map

| Search / heuristic | Moves | Cost | States expanded/evaluated |
|---|---:|---:|---:|
| BFS | 40 | 40 | 64 |
| A*, (h=0) | 40 | 40 | 64 |
| A*, Euclidean | 40 | 40 | 64 |
| A*, Manhattan | 40 | 40 | 64 |
| A*, (2h_M) | 40 | 40 | 67 |
| A*, (5h_M) | 40 | 40 | 84 |

On this particular map, Manhattan A* ties BFS: both return a 40-move optimal path and evaluate 64 states. It would be inaccurate to claim an expansion advantage here. The two inflated variants also happen to return the optimal length on this map, but that single observation does not establish a general guarantee.

### Branching alternative-path map

| Search / heuristic | Moves | Cost | States expanded/evaluated |
|---|---:|---:|---:|
| BFS | 7 | 7 | 18 |
| A*, (h=0) | 7 | 7 | 18 |
| A*, Euclidean | 7 | 7 | 14 |
| A*, Manhattan | 7 | 7 | 14 |
| A*, (2h_M) | 7 | 7 | 14 |
| A*, (5h_M) | 9 | 9 | 11 |

This map makes the trade-off visible. Admissible Euclidean and Manhattan A* find the same 7-move optimum as BFS while evaluating 14 rather than 18 states. The (5h_M) variant evaluates only 11 states but returns a valid 9-move route, which is not optimal. The (2h_M) variant happens to return the optimum on this small map; because the heuristic is inflated, that result is empirical rather than a general guarantee.

### Edge-case maps

- The trivial map returns a one-move path; the measured count is two evaluated states (start and goal).
- The no-solution map correctly returns no path after evaluating nine states.
- The branching map checks that the solver does not confuse a valid route with a shortest route.

The raw paths and outputs are retained in `search_output.txt`; the independent checks are retained in `test_results.txt`.

## 5. LLM prompt and evaluation

The following prompt was used in this ChatGPT session to obtain an independent conceptual review of the heuristic design:

> Review a four-connected grid pathfinding problem with unit-cost moves. Compare BFS with A* using zero, Euclidean, Manhattan, 2×Manhattan, and 5×Manhattan heuristics. State which heuristics are admissible and consistent, explain when optimality is guaranteed, and propose tests that would expose a heuristic that expands fewer states but returns a longer path. Do not infer optimality from expansion count alone.

### Review summary

- BFS is optimal for this problem because all legal actions have equal cost.
- (h=0) makes A* equivalent in priority principle to uniform-cost search; with unit costs this has the same optimality result as BFS.
- Manhattan distance is the natural exact obstacle-free lower bound for four-connected unit-cost movement. Euclidean distance is also a lower bound, though it can be less informed.
- Scaling Manhattan by a factor greater than 1 can overestimate the true remaining cost. Weighted A* may find a solution with fewer expansions, but ordinary A*'s optimality guarantee no longer follows.
- A useful evaluation must include both an open/branching map and edge cases such as a one-step goal and an unreachable goal. It must validate the returned path and compare its cost with BFS.

### Comparison against the implementation

The measured branching-map result supports the warning: (5h_M) returns a valid 9-move path instead of the 7-move BFS optimum, despite evaluating fewer states (11 versus 18). The original map alone would have concealed this issue because all five heuristic variants returned 40 moves there. This is why the alternative-path map is necessary.

**Disclosure:** The prompt and review summary above were generated in this ChatGPT session. They are not represented as an output from a separately accessed model or a third-party LLM. No external model transcript was supplied.

## 6. Reflection and limitations

1. A* is not automatically faster than BFS on every map. On the supplied warehouse map, Manhattan A* and BFS have identical measured path length and expansion count.
2. Heuristic quality depends on the topology. The branching map shows a measurable expansion reduction for admissible heuristics.
3. Weighted heuristics trade guarantees for search bias. The (5h_M) result is faster by the measured expansion count but gives a longer route.
4. Expansion counts need a declared convention. This implementation counts the terminal goal consistently for BFS and A*. Under that convention, the branching-map BFS count is 18; an older count of 17 omitted the terminal goal.
5. Tests should verify semantics, not just whether a path exists: start and goal, legal adjacent steps, wall avoidance, path cost, optimality against BFS, malformed inputs, and unreachable cases all matter.

## 7. Reproduction

From this directory:

```bash
python3 test_search.py
python3 search_agent.py
```

The current independent suite contains nine checks. The measured results are recorded in `test_results.txt` and `search_output.txt`.
