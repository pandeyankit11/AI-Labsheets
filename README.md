# AI Labsheets

Laboratory question sheets, implementations, independent tests, generated evidence, and written answers for the Artificial Intelligence course.

## 01. Bayesian Networks and Autoregressive Language Models

Directory: `01_bayesian_networks/`

- First-order and second-order autoregressive language models.
- Conditional probability tables, normalization checks, parameter accounting, context sparsity, and generation diversity.
- Evidence: `answers.md`, `lab_output.txt`, `generated_examples.txt`, `test_results.txt`, and `reflection.md`.

## 02. Logical Planning

Directory: `02_logic_hands_on/`

- Original warehouse STRIPS planner and optional Prolog verifier.
- Revised locked-door domain with six actions, shortest-plan BFS, impossible case, first-error plan validation, and LLM-plan repair comparison.
- Evidence: `locked_door_results.md`, `planner_output.txt`, and `test_results.txt`.

## 03. Search and A*

Directory: `03_search_hands_on/`

- The Search question sheet is `AI_Lab_Search_AStar_Submission.docx` (Tasks 0–9).
- BFS and A* on the original warehouse map and additional edge-case/alternative-route maps.
- Exact start/goal marker validation and consistent expansion-event counting.
- Five heuristic variants on the original and branching maps.
- Written answers and LLM prompt/evaluation: `search_answers.md`.
- Evidence: `heuristic_experiments.md`, `search_output.txt`, and `test_results.txt`.

## Run the checks

```bash
cd 01_bayesian_networks
python3 -m unittest -v test_language_models.py
python3 run_lab.py

cd ../02_logic_hands_on
python3 test_planner.py
python3 planner.py

cd ../03_search_hands_on
python3 test_search.py
python3 search_agent.py
```

`verification_results.txt` records the test counts and measured outcomes. `TASK_AUDIT.md` maps each available question sheet to implementations and deliverables.

The Search DOCX question sheet and original Logic submission report are preserved unchanged.
