# Reinforcement Learning and Deep RL Lab

This folder contains a question-sheet-aligned implementation of the CS-F407 RL laboratory.

## Contents
- RL_lab_2026.pdf: original 13-page question sheet.
- rl_core.py: single/two-robot environments, tabular Q-learning, DQN networks, replay training, and evaluation helpers.
- tabular_q_learning_implementation.py: Tasks 1-5.
- dqn_single_robot.py: Task 6.
- dqn_two_robots.py: Tasks 7-8.
- test_rl_lab.py: deterministic environment, collision, TD target, truncation, encoding, and architecture tests.
- lab_report_submission.md: report with measured outcomes and reflection answers.
- llm_prompt.txt: retained implementation specification / reproducibility prompt.
- test_results.txt: verification summary.
- results/: machine-readable results, moving-average plots, and actual greedy trajectory.

## Run
python -m pip install -r requirements.txt
python -m unittest -v test_rl_lab.py
python tabular_q_learning_implementation.py
python dqn_single_robot.py
python dqn_two_robots.py

The DQN code calls torch.set_num_threads(1) before model initialization to reduce CPU oversubscription for small networks. This does not change the prescribed network, rewards, replay settings, or training budgets.

The tabular and one-robot DQN greedy policies reach the goal in four moves from the default start and succeed on the required additional start. The two-robot DQN actual learned trajectory is saved. The manual collision-free route in the report is clearly distinguished from the learned DQN rollout.
