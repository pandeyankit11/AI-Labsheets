# AI Labsheets

Laboratory assignments, implementations, independent tests, generated evidence, and submission supplements for the Artificial Intelligence course.

## 01. Bayesian Networks and Autoregressive Language Models

Directory: `01_bayesian_networks/`

- First-order and second-order language models with sampling and greedy generation.
- Probability normalization tests.
- Corrected second-order parameter count, including the initial-token distribution.
- Explicit second-order context universe and zero-context count.
- Evidence: `answers.md`, `lab_output.txt`, `generated_examples.txt`, and `reflection.md`.

## 02. Logical Planning

Directory: `02_logic_hands_on/`

- Original warehouse STRIPS planner retained.
- Revised locked-door/key domain with six actions.
- BFS shortest plan: `PickKey -> Move(A,B) -> Unlock -> Move(B,C)`.
- Impossible case without `PickKey`.
- `validate_plan()` reports the first unsatisfied precondition in a proposed plan.
- Supplement: `locked_door_results.md`.

## 03. Search and A*

Directory: `03_search_hands_on/`

- BFS and A* on the original warehouse map and three test maps.
- Consistent expansion-event counting and exact `S`/`G` map-marker validation.
- Experiments for `h=0`, Euclidean distance, and `2 x Manhattan`.
- Supplement: `heuristic_experiments.md`.

## Run all checks

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

The checked outputs are summarized in `verification_results.txt`. Local verification recorded 7 Bayesian unit tests, 8 planning checks, and 7 search checks passing.

The original DOCX submission files are retained unchanged. The revised planning and heuristic-experiment evidence has been added as adjacent Markdown supplements; those DOCX reports still need to be updated before treating the full formatted submission package as final.
