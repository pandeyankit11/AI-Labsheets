# AI Labsheets: Source-to-Deliverable Audit

Audit date: 25 November 2026  
Branch: `complete-remaining-lab-tasks`

This audit is based on the three supplied PDF lab sheets and the Search Word question sheet, `03_search_hands_on/AI_Lab_Search_AStar_Submission.docx`. The Search DOCX is the source handout for Tasks 0–9, not a completed submission report. The original Logic DOCX submission report was inspected separately.

## 01 — Bayesian Networks (`lab_sheet.pdf`, 11 pages)

| Sheet requirement | Evidence/status |
|---|---|
| Q1–Q14: chain rule, first-order conditional independence, CPTs, generation, second-order model, modern autoregressive connection, LLM reflection | Covered in `answers.md` |
| First-order and second-order implementations | `language_models.py` |
| Selected conditional probability tables and normalization checks | `lab_output.txt` |
| At least 20 sampled outputs | 20 per model in `generated_examples.txt` |
| Five greedy runs per model | Five per model are recorded. Second-order terminates. First-order strict greedy cycles on this corpus and each output is capped at 12 tokens; these are explicitly labeled as prefixes, not complete sentences. |
| Parameter count and zero-probability contexts | `answers.md` defines the context universe: 110 possible, 14 observed, 96 unobserved. Parameter entries: 17 first-order and 19 second-order including the separate initial-token distribution. |
| Diversity/coherence comparison | 20 sampled outputs saved; 10 unique first-order outputs and 6 unique second-order outputs. Greedy produces one unique output per model because it is deterministic. |
| Reflection includes code inspected/corrected | `reflection.md` contains the parameter-count correction example and explains the omitted initial-token distribution. |

**Assessment:** Implementation and most deliverables are complete. The first-order greedy special case cannot produce five completed sentences while preserving strict argmax decoding, because the most-probable transitions form a cycle. This is disclosed instead of forcing an end token or choosing a lower-probability transition.

## 02 — Original Logical Planning (`logic_lab_ex.pdf`, 14 pages)

The existing `AI_Lab_Logical_Planning_Submission.docx` covers the original sheet's Tasks 0–5, original tests A/B/C, the logic/search explanation, reflection answers, and optional Prolog Tasks 6–8. It records the correct warehouse plan and full state trace and explicitly identifies the handout's illustrative Move-before-PickUp ordering as invalid. The Python and Prolog source files are retained.

**Assessment:** Original-sheet written submission is already covered in the DOCX; no rewrite was needed.

## 03 — Revised Logical Planning (`planning_lab_revised.pdf`, 16 pages)

| Sheet requirement | Evidence/status |
|---|---|
| Task 6 exact six-action domain using `At(A/B/C)`, `KeyAt(A)`, `HasKey`, `Unlocked` | Implemented in `planner.py` using the exact propositions and action preconditions/effects |
| Exactly one `At(·)` proposition in each reachable state | Checked by `exactly_one_robot_location()` and tests |
| Task 6(a)/(b) shortest plan and complete state trace | Four-action BFS plan and `S0`–`S4` trace in `planner.py`; documented in `locked_door_results.md` |
| Task 6(c), remove `PickKey` | BFS returns no plan; test and report included |
| Task 6(d), validate `Move(A,B), Unlock, Move(B,C)` | First failure is step 2 `Unlock`, missing `HasKey`; test and report included |
| Task 7 candidate, validator, feedback, revision, goal/length comparison | Documented in `locked_door_results.md`; initial candidate is rejected, revised candidate validates and matches BFS length 4 |
| Revised Prolog extension | Explicitly optional; the original optional Prolog evidence remains in the DOCX |

**Assessment:** Required revised-domain tasks are implemented and documented. The LLM comparison is labeled as the candidate/revision generated in this ChatGPT session, not falsely attributed to another queried model.

## 04 — Search & A* (`AI_Lab_Search_AStar_Submission.docx`, Tasks 0–9)

The Search DOCX is the question sheet and is present in `03_search_hands_on/`. It was audited as the source for Tasks 0–9. It is not a missing PDF, and the absence of a separate Search PDF is not a blocker.

| Question-sheet area | Evidence/status |
|---|---|
| Problem formulation: state, initial state, actions, transition model, goal test, and path cost | Defined in `search_answers.md` and implemented in `search_agent.py` |
| Search design and BFS/A* behavior | Explained in `search_answers.md`; implemented with path reconstruction in `search_agent.py` |
| Map-based execution and path/cost/expansion comparison | Original warehouse and additional maps are retained; measured output is in `search_output.txt` |
| Edge cases and independent validation | Trivial, unreachable, and alternative-route maps; nine independent checks in `test_search.py` and `test_results.txt` |
| Heuristic investigation | Five variants on the original and branching maps; tables and interpretation in `heuristic_experiments.md` and `search_answers.md` |
| LLM prompt, evaluation, and reflection | Prompt, review summary, comparison with measured behavior, and limitations in `search_answers.md`. The LLM review is explicitly disclosed as generated in this ChatGPT session, not an external-model transcript. |
| Expansion-count consistency | BFS/A* count evaluated frontier removals including the goal; branching-map BFS count is 18 under this convention, superseding an older count of 17 that omitted the terminal goal. |
| Admissibility/optimality boundary | Branching map shows `5× Manhattan` returns a valid but non-shortest 9-move path where the optimum is 7. |

**Assessment:** The Search question sheet has been source-audited. Implementation, edge-case tests, heuristic experiments, written answers, and the LLM-prompt/evaluation section are present. The LLM section does not pretend that a separate third-party model was queried.

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
