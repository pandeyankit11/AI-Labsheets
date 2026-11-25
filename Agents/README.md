# Goal-Based Warehouse Navigation Agent

This folder contains the completed **Agents: Laboratory Exercise** submission. The original question sheet, `agents_lab.pdf`, is preserved unchanged.

## Files

| File | Purpose |
|---|---|
| `goal_based_agent.py` | Goal-based agent and BFS planner; prints the selected path and route map. |
| `warehouse_map.txt` | Exact 7 × 21 warehouse map from the question sheet. |
| `test_goal_based_agent.py` | 20 unit tests for parsing, movement, path validity, optimality, and failure cases. |
| `REPORT.md` | Answers to Tasks 1–3, design, algorithm justification, results, and limitations. |
| `PROMPT_ENGINEERING.md` | Initial/refined LLM prompts and the iteration/evaluation record. |
| `agent_architecture.svg` | Block diagram of the goal-based agent architecture. |
| `lab_output.txt` | Captured output from running the supplied map. |
| `test_results.txt` | Captured test-run output. |

## Run

Requires Python 3.9 or newer. The implementation uses only the Python standard library.

```bash
cd Agents
python3 goal_based_agent.py
python3 -m unittest -v test_goal_based_agent.py
```

The planner uses Breadth-First Search (BFS). Since every legal move costs one step, BFS is complete on this finite grid and returns a shortest route whenever one exists.

## Result on the supplied map

- Grid: 7 rows × 21 columns
- Start: `(1, 1)` (zero-based row, column)
- Goal: `(1, 19)`
- Shortest route: **20 moves**
- Test suite: **20 passed, 0 failed**

See `REPORT.md` for the full answer and `PROMPT_ENGINEERING.md` for the LLM-assisted development record.
