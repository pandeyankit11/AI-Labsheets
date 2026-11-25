# AI Labsheets: Source-to-Deliverable Audit

Audit date: 25 November 2026
Branch: complete-remaining-lab-tasks

This audit covers the original lab sheets stored in the repository: Bayesian Networks PDF, original and revised Logic PDFs, Search DOCX handout, Neural Models PDF, and Reinforcement Learning PDF. The Search DOCX is the source handout for Tasks 0-9, not a completed submission report.

## 01 — Bayesian Networks
See the detailed Bayesian section already present in the earlier audit. Implementation, CPTs, generation outputs, parameter counting, and reflection are present. The first-order strict-greedy non-termination on this corpus is explicitly disclosed rather than forcing a non-argmax end token.

## 02 — Logical Planning
The original Logic DOCX covers original Tasks 0-5 and optional Prolog evidence. The revised locked-door Tasks 6-7 are implemented and documented, including the four-action BFS plan, impossible no-key case, first failed precondition, and revised candidate validation.

## 03 — Search and A*
Source question sheet: 03_search_hands_on/AI_Lab_Search_AStar_Submission.docx, Tasks 0-9.
Implementation, path reconstruction, alternate maps, five heuristic variants, edge cases, measured expansion counts, written answers, prompt, and disclosed LLM review are present. The branching-map weighted heuristic result is valid but non-optimal (9 moves versus the 7-move optimum).

## 04 — Neural Models (04_neural_models/lab_sheet.pdf, 7 pages)

| Source requirement | Deliverable and verification |
|---|---|
| Task 1: XOR input/output spaces, four labels, point layout, linear-separability explanation, linear-model prediction | REPORT.md; coordinate sketch added. Linear baseline result is loss about 0.693147, 2/4 correct. |
| Task 2: 2-2-1 design, nonlinear hidden activation, binary output/loss pair, success criteria | xor_lab.py and REPORT.md; BCEWithLogitsLoss used with a single binary logit. |
| Task 3: LLM implementation prompt and code inspection | llm_prompt.txt, xor_lab.py, REPORT.md; prompt is disclosed as a retained specification, not an external-model transcript. |
| Task 4A: loss, four probabilities, labels | results/activation_results.json and lab_output.txt; sigmoid/tanh/ReLU each classify all four inputs correctly. |
| Task 4B: actual first-layer gradient after backward and mean-loss interpretation | Added capture of the actual 2x2 fc1.weight.grad before optimizer step, saved in JSON/CSV/text, with values in REPORT.md. Matrix norms were independently recomputed and match scalar norms within floating-point tolerance. |
| Task 4C: all-zero symmetry experiment | results/symmetry_results.json; hidden rows remain identical, first-layer gradient norm remains zero, loss remains 0.693147. |
| Task 4D: sigmoid/tanh/ReLU comparison and early gradient norms | results/activation_results.json, CSV, and REPORT.md; same seed/initial weights reused so activation is the changed factor. |
| Task 5: three-class logits/CrossEntropy, probabilities, normalization, p-y gradient, +100 shift diagnostic | results/three_class_results.json, lab_output.txt, REPORT.md; 2-2-3 output, all predictions correct, probability sums approximately 1, shift error approximately 1.03e-9. |
| Reflection Q1-Q7 | REPORT.md. |

Neural limitation: a regression test for saved gradient tensors was added to test_lab.py, but the full Neural unittest suite was not executed in this audit runtime. The matrix values and norms were independently checked numerically. REPORT.md is authoritative; report.pdf was not regenerated to include the gradient appendix.

## 05 — Reinforcement Learning (RL_lab/RL_lab_2026.pdf, 13 pages)

| Source requirement | Deliverable and verification |
|---|---|
| Task 1: MDP, fixed grid, actions, rewards, initial/terminal state, shortest route | rl_core.py and lab_report_submission.md; cell (2,2) is an open doorway, so shortest route is 4 moves, not the earlier draft's 8. |
| Task 2-3: Q-table design, epsilon-greedy, seed/settings, prompt | tabular_q_learning_implementation.py, llm_prompt.txt, and report. |
| Task 4: environment checks and manual TD updates | test_rl_lab.py; source's nonterminal reward is -1, giving target 0.9 and updated Q 0.5. Terminal +5 gives 1.32; truncation bootstraps. |
| Task 5: training curves, greedy evaluation, random baseline, additional start | results/q_learning_results.json and q-learning SVGs; default succeeds in 4 moves and additional start (0,0) in 6; random baseline has 15% success in 100 episodes. |
| Task 6: one-hot 25 input, 64-64-5 DQN, replay, target network, terminal/truncation targets, sync schedule, 12k steps | dqn_single_robot.py, rl_core.py, and results/single_robot_dqn_results.json; seed set before network creation; target synchronization counts optimizer updates. Frozen greedy route succeeds in 4 default steps and 6 additional-start steps. |
| Task 7: 50 input/25 joint output, simultaneous transitions, goal occupancy, same-cell/swap/wait/follow collision rules and team reward | TwoRobotWarehouseEnv in rl_core.py and 18-test suite; explicit tests distinguish same-cell from swap, entry into a waiting robot, and following into a vacated cell. |
| Task 8: compare three controllers, additional starts, reflection | results JSON files, trajectory SVG, and lab_report_submission.md; frozen greedy two-robot evaluation succeeds in 9 joint steps from default and 10 from A=(0,0), B=(4,4). |

RL limitations: two-robot DQN training has a 28.90% episode success rate, 28.53% collision rate, and 42.57% timeout rate during exploration; the final 50 training episodes have 96% success. The 9-/10-step results are deterministic frozen-greedy evaluations from the two starts specified by the sheet, not broad policy generalization. Manual routes are distinguished from the learned policy.

Verification: python -m unittest -v test_rl_lab.py returned 18 tests passed; python -m compileall -q . succeeded. The suite covers movement, open doorway, rewards, terminal/truncation, TD arithmetic, epsilon schedule, joint-action encoding, required collision cases, goal occupancy, target masking, and network output dimensions.

## Final assessment

- Bayesian: implemented and documented, with first-order strict-greedy non-termination transparently disclosed.
- Original Logic: covered by original submission DOCX and code.
- Revised Logic: required Tasks 6-7 implemented and documented.
- Search: source-audited against DOCX handout; implementation, test evidence, heuristic analysis, written answers, and LLM-review disclosure present.
- Neural Models: code and report cover Tasks 1-5; actual gradient tensor evidence added; full automated suite still requires execution in a local Python/PyTorch environment.
- RL: corrected implementations and report pushed; 18 targeted tests pass locally; two-robot policy succeeds on the two prescribed greedy evaluation starts, while exploration failures remain disclosed.

All updates are on complete-remaining-lab-tasks. The main branch is not modified.
