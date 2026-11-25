# A* Heuristic Experiments

All experiments use the checked-in `warehouse_map.txt`, unit move cost, the same neighbor order, and the same A* implementation.

## Results

| Heuristic | Heuristic weight | Path length (moves) | Path cost | States expanded/evaluated |
|---|---:|---:|---:|---:|
| `h = 0` (uniform-cost search) | 1 | 40 | 40 | 64 |
| Euclidean distance | 1 | 40 | 40 | 64 |
| `2 x Manhattan distance` | 1 (heuristic is scaled) | 40 | 40 | 67 |

The independently checked shortest path is 40 moves. The weighted heuristic also found a 40-move path on this specific map, but multiplying an admissible heuristic by 2 can overestimate the remaining cost, so optimality is not guaranteed in general.

The baseline comparison is intentionally not presented as an A* expansion win: Manhattan A* and BFS both evaluate 64 states on this map. The zero and Euclidean experiments also evaluate 64. The weighted-Manhattan run evaluates 67.

## Expansion-count convention

`expanded_states` counts each non-stale state removed from the frontier and evaluated, including the terminal goal. BFS marks states visited when enqueuing them, so duplicates are not expanded. A* ignores stale heap entries and can reopen a previously expanded state if a strictly better `g`-score is found; each such re-expansion counts as another event. The goal is counted when selected from the frontier, but its successors are not generated.

## Reproduction

```bash
python3 test_search.py
python3 search_agent.py
```

The suite checks the 40-move original map, trivial map, unreachable map, 7-move alternative-path map, exact start/goal marker counts, and the three requested heuristic experiments.
