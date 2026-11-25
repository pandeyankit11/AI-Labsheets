# AI Labsheets: Source-to-Deliverable Audit

Audit date: 25 November 2026  
Branch: `complete-remaining-lab-tasks`

This audit is based on the PDFs present in the current branch and the two existing DOCX submission reports. The original Word reports were inspected rather than assumed complete from their filenames.

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

## 04 — Search & A*

**Source-file limitation:** The current recursive GitHub tree contains no Search lab question-sheet PDF in `03_search_hands_on/` (nor a file named as a Search lab sheet). The detailed `AI_Lab_Search_AStar_Submission.docx` exists and was inspected; it documents the original Search exercise Tasks 0–9, including formulation, design, prompt, BFS/A*, maps, test results, heuristic investigation, LLM evaluation, and reflection. However, that report is a submission report, not the original question sheet. Exact source-to-task verification against the Search question-sheet PDF remains unconfirmed until that PDF is present in the branch.

| Requirement evidenced by the existing Search report/code | Evidence/status |
|---|---|
| Search formulation, state/action/transition/cost, design and LLM prompt | Existing DOCX report |
| BFS/A* implementations, path reconstruction, maps and independent tests | `search_agent.py`, `test_search.py`, map files and existing DOCX |
| Heuristic experiments | Five variants (`h=0`, Euclidean, Manhattan, `2×`, `5×`) on both original and branching maps; results in `heuristic_experiments.md` |
| Expansion-count consistency | BFS/A* count evaluated frontier removals including the goal; branching-map BFS count is 18 under this convention, superseding old DOCX count 17 |
| Exactly one start and goal | `parse_map()` enforces this and tests duplicate/missing markers |
| Admissibility/optimality boundary | Branching map shows `5× Manhattan` returns a valid but non-shortest 9-move path where the optimum is 7 |

**Assessment:** Implementation and experiments are strengthened against the available Search report. Exact verification against the original Search question-sheet PDF cannot be certified because the PDF is absent from the branch.

## Final status

- Bayesian: substantially complete, with first-order greedy's non-terminating argmax loop explicitly disclosed.
- Original Logic: complete according to the existing DOCX report and code.
- Revised Logic: required Tasks 6–7 complete in code, tests, and supplement.
- Search: implementation and available report requirements audited; original question-sheet verification remains blocked by the absent source PDF.

Latest rerun: 7 Bayesian tests, 12 planning checks, and 9 Search checks passed; `compileall` succeeded for all three directories. Raw test and demonstration outputs are saved beside the relevant lab code.

No original DOCX report has been overwritten. New and corrected source, test, and Markdown evidence is committed on `complete-remaining-lab-tasks`; `main` is not changed.
