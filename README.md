# AI Labsheets

Laboratory question sheets, implementations, independent tests, generated evidence, and written answers for the Artificial Intelligence course.

## 01. Bayesian Networks and Autoregressive Language Models
Directory: 01_bayesian_networks/

- First-order and second-order autoregressive language models.
- Conditional probability tables, normalization checks, parameter accounting, context sparsity, and generation diversity.
- Evidence: answers.md, lab_output.txt, generated_examples.txt, test_results.txt, and reflection.md.

## 02. Logical Planning
Directory: 02_logic_hands_on/

- Original warehouse STRIPS planner and optional Prolog verifier.
- Revised locked-door domain with six actions, shortest-plan BFS, impossible case, first-error plan validation, and LLM-plan repair comparison.
- Evidence: locked_door_results.md, planner_output.txt, and test_results.txt.

## 03. Search and A*
Directory: 03_search_hands_on/

- The Search question sheet is AI_Lab_Search_AStar_Submission.docx (Tasks 0–9).
- BFS and A* on original and alternative warehouse maps.
- Five heuristic variants, edge-case tests, written answers, prompt, and disclosed LLM review.
- Evidence: search_answers.md, heuristic_experiments.md, search_output.txt, and test_results.txt.

## 04. Neural Models
Directory: 04_neural_models/

- XOR linear baseline and nonlinear 2-2-1 models with sigmoid, tanh, and ReLU.
- Zero-initialization symmetry, actual first-layer gradient tensors, and three-class CrossEntropyLoss extension.
- Evidence: REPORT.md, results/activation_results.json, results/gradient_tensors.json, results/symmetry_results.json, and results/three_class_results.json.
- REPORT.md is authoritative. The older report.pdf snapshot was not regenerated for the gradient-tensor appendix.

## 05. Reinforcement Learning and Deep RL
Directory: RL_lab/

- Tabular Q-learning, single-robot DQN, and joint two-robot DQN.
- Correct termination-vs-truncation bootstrapping, fixed seed timing, target sync counted by optimizer updates, and collision-rule tests.
- Evidence: lab_report_submission.md, test_rl_lab.py, test_results.txt, and results/ JSON/SVG artifacts.

## 06. Goal-Based Agents
Directory: Agents/

- Goal-based warehouse navigation agent using Breadth-First Search (BFS).
- Exact 7 × 21 map from the Agents question sheet, path reconstruction, named actions, and route visualization.
- Task 1 problem analysis, Task 2 design and architecture diagram, Task 3 initial/refined prompts and LLM evaluation.
- Evidence: REPORT.md, PROMPT_ENGINEERING.md, agent_architecture.svg, lab_output.txt, and test_results.txt.

## Run the checks

    cd 01_bayesian_networks && python3 -m unittest -v test_language_models.py
    cd ../02_logic_hands_on && python3 test_planner.py
    cd ../03_search_hands_on && python3 test_search.py
    cd ../04_neural_models && python3 -m unittest -v test_lab.py
    cd ../RL_lab && python3 -m unittest -v test_rl_lab.py
    cd ../Agents && python3 -m unittest -v test_goal_based_agent.py

verification_results.txt records test counts and measured outcomes. TASK_AUDIT.md maps available question sheets to implementations and deliverables. The original question sheets are preserved unchanged.
