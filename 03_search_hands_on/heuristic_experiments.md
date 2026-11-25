# A* Heuristic Experiments and Expansion-Count Audit

The source handout is `AI_Lab_Search_AStar_Submission.docx` (Tasks 0–9) in this directory. It is the Search question sheet, not a prior submission report. The written answers, problem formulation, BFS/A* explanation, LLM prompt/evaluation, and reflection are in `search_answers.md`. This supplement records the measured heuristic experiments using the current implementation and one consistent expansion-count definition.

## Experimental setup

- Moves are four-connected (`Up`, `Down`, `Left`, `Right`), each with cost 1.
- The primary heuristic sweep uses the original `warehouse_map.txt`. A second sweep uses `map_altpaths.txt`, which has genuine route alternatives and makes heuristic differences visible.
- The five tested variants are `h=0`, Euclidean distance, Manhattan distance, `2 × Manhattan`, and `5 × Manhattan`.
- The inflated Manhattan variants are weighted/inadmissible in general. Their observed path lengths are measured rather than assuming that they remain optimal.

## Original warehouse map

| Search / heuristic | Path length (moves) | Path cost | States expanded/evaluated |
|---|---:|---:|---:|
| BFS | 40 | 40 | 64 |
| `h = 0` | 40 | 40 | 64 |
| Euclidean | 40 | 40 | 64 |
| Manhattan | 40 | 40 | 64 |
| `2 × Manhattan` | 40 | 40 | 67 |
| `5 × Manhattan` | 40 | 40 | 84 |

The measured result is that Manhattan A* and BFS tie on the original map: both find 40 moves and evaluate 64 states. This map offers no A* expansion advantage under this implementation. That tie is reported as measured, not turned into a fictional A* win.

## Branching alternative-path map

| Search / heuristic | Path length (moves) | Path cost | States expanded/evaluated |
|---|---:|---:|---:|
| BFS | 7 | 7 | 18 |
| `h = 0` | 7 | 7 | 18 |
| Euclidean | 7 | 7 | 14 |
| Manhattan | 7 | 7 | 14 |
| `2 × Manhattan` | 7 | 7 | 14 |
| `5 × Manhattan` | 9 | 9 | 11 |

The branching map demonstrates a useful difference: admissible Euclidean/Manhattan A* expands 14 states versus BFS's 18 while preserving the 7-move optimum. The `5 × Manhattan` variant expands only 11 states but returns a valid 9-move path, not the shortest path. This shows that an aggressive overestimate can reduce work while invalidating A*'s optimality guarantee. The `2 × Manhattan` variant happens to return the 7-move optimum on this small map, but that empirical result is not a general guarantee because the scaled heuristic is not admissible in general.

## Expansion-count convention

`expanded_states` counts each non-stale state removed from a frontier and evaluated, including the terminal goal. BFS marks a state visited when it is enqueued, so duplicate states are not expanded. A* ignores stale heap entries and can reopen an already-expanded state if a strictly lower `g`-score is found; each such re-expansion counts as another expansion event. The goal is counted when popped, but its successors are not generated.

Under this explicit convention, the branching-map BFS count is 18, not 17 as stated in the older question-sheet material. That older count omitted the terminal goal in BFS's count. The present code and tests use one convention for both algorithms; this supplement supersedes the older expansion-count number without modifying the question sheet.

## Other map checks

| Map | Expected result | Current verified result |
|---|---|---|
| Trivial | One move | One move; 2 states evaluated including the goal |
| No-solution | No path | No path; 9 states evaluated |
| Alternative paths | Shortest path of 7 moves | BFS and Manhattan A* both return 7 moves |

`parse_map()` rejects maps with missing or duplicate `S` or `G` markers and unsupported symbols.

## Reproduction

```bash
python3 test_search.py
python3 search_agent.py
```

The test suite exercises both maps for all five heuristic variants, validates exact start/goal counts, checks path validity, and verifies consistent expansion accounting. Captured output is saved in `search_output.txt` and `test_results.txt`.
