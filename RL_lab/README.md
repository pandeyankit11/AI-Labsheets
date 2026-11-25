# Reinforcement Learning and Deep RL Lab

Question sheet: RL_lab_2026.pdf (13 pages).

## Files
- rl_core.py: tested grid environments, Q-learning update, DQN replay/target calculations, and evaluation.
- tabular_q_learning_implementation.py: Tasks 1-5.
- dqn_single_robot.py: Task 6.
- dqn_two_robots.py: Tasks 7-8.
- test_rl_lab.py: 18 environment, TD-update, truncation, collision, and network-shape checks.
- lab_report_submission.md: corrected, question-sheet-aligned report and measured outcomes.
- llm_prompt.txt: retained implementation specification.
- requirements.txt: runtime dependencies.
- test_results.txt: verification summary.
- results/: measured experiment JSON summaries and SVG moving-average/trajectory plots. Re-running the scripts also generates detailed episode CSVs, the Q-table CSV, and PNG copies of the plots.

## Run

    python -m pip install -r requirements.txt
    python -m unittest -v test_rl_lab.py
    python tabular_q_learning_implementation.py
    python dqn_single_robot.py
    python dqn_two_robots.py

The DQN code seeds all RNGs before constructing networks and syncs the target network after each 250 optimizer updates. It uses one PyTorch CPU thread to avoid oversubscribing these small networks.

The tabular and one-robot DQN greedy policies reach the goal in four moves from the default start and succeed on the required additional start. The two-robot learned greedy paths are saved as JSON and SVG. Training still includes exploration-induced collisions and timeouts, reported separately from greedy evaluation.
