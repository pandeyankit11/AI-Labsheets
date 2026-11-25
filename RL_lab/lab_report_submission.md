# Reinforcement Learning and Deep RL Lab Report

Course: CS-F407 Artificial Intelligence  
Source: RL_lab_2026.pdf (13 pages)  
Verification environment: Python 3.13.5, NumPy 2.3.5, Matplotlib 3.10.8, PyTorch 2.10.0+cpu  
Seeds: 7 for learned agents; random baseline seed 17.

This report separates the sheet requirements, implementation, and measured results. A manually designed route is not presented as a learned DQN trajectory.

## 1. Part I - Tabular Q-learning

### Task 1: MDP specification

| Component | Specification |
|---|---|
| State | Current coordinate (row,column); the map and goal are fixed |
| State encoding | 5*row+column; 25 rows in the Q table |
| Actions | 0 Up, 1 Down, 2 Left, 3 Right, 4 Wait |
| Obstacles | {(0,2),(1,2),(3,2),(4,2)} |
| Start / goal | S=(2,0), G=(2,4) |
| Transition | Deterministic; blocked boundary/obstacle attempt leaves the robot in place |
| Reward | +10 at goal; -1 for invalid move; -0.1 for any other valid action including Wait |
| True terminal | Reaching (2,4) |
| Truncation | Unfinished episode stopped after 40 steps; not true termination |
| Discount | gamma=0.95 |

The centre cell (2,2) is the open doorway, not an obstacle. The shortest route is (2,0)->(2,1)->(2,2)->(2,3)->(2,4), length 4. An earlier eight-step route was wrong because it ignored the doorway. The corrected route is tested.

Current position is Markov because the map/goal are fixed and transitions/rewards are deterministic. Search uses a known model to plan a route; Q-learning estimates action values from sampled reward transitions.

### Task 2: Agent design

The agent uses a zero-initialized 25x5 Q table. The update is:
- y=r for true terminal transitions;
- otherwise y=r+gamma*max_a' Q(s',a');
- Q(s,a) <- Q(s,a)+alpha*(y-Q(s,a)).

Settings: alpha=0.2, gamma=0.95, seed 7, 1,500 episodes, epsilon decayed from 1.0 to 0.05 over the first 1,200 episodes. Training ties are random; frozen greedy evaluation uses deterministic first-maximum tie-breaking and no updates. Return, success, timeout, invalid moves, and epsilon are recorded. Training plots use 50-episode moving averages.

### Task 3: Implementation prompt and code review

The reproducibility prompt/specification is stored in llm_prompt.txt. It records the map, rewards, termination/truncation distinction, Q-learning schedule, DQN settings, and validation constraints. It is not represented as a verbatim transcript of an unavailable external model chat.

### Task 4: Environment and update tests

| Start | Action | Result | Reward | Terminal? |
|---|---|---|---:|---|
| (0,0) | Up | Remains (0,0), boundary | -1 | No |
| (1,1) | Right | Remains (1,1), obstacle at (1,2) | -1 | No |
| (2,1) | Right | Moves into doorway (2,2) | -0.1 | No |
| (2,1) | Wait | Remains (2,1), valid wait | -0.1 | No |
| (2,3) | Right | Reaches (2,4) | +10 | Yes |
| Non-goal cell | Step 40 | Time limit | Per action | Truncated |

The sheet's synthetic update has Q=0.4, r=-1, max next Q=2, alpha=0.2, gamma=0.95:
- target y=-1+0.95*2=0.9;
- updated Q=0.4+0.2*(0.9-0.4)=0.5.

For true terminal reward +5, next-state value is ignored: y=5 and updated Q=1.32. For truncation, the pre-reset next state is bootstrapped, so target/update are 0.9/0.5. Only true termination masks bootstrap. An earlier draft used +1; the source sheet specifies -1.

### Task 5: Training and evaluation

| Measurement | Result |
|---|---:|
| Episodes / seed | 1,500 / 7 |
| alpha, gamma | 0.2, 0.95 |
| Epsilon | 1.0 initially; 0.05 by episode 1,200 |
| Final 50-episode success | 100% |
| Final 50-episode mean return | 9.654 |

Frozen greedy results:
- Default start (2,0): success in 4 steps, zero invalid moves, direct doorway path.
- Additional start (0,0): success in 6 steps, zero invalid moves, path (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)->(2,4).

Uniform-random baseline (100 episodes, seed 17): success 15%, timeout 85%, mean successful episode length 22.6, mean invalid moves 9.66. Training return can remain noisy due to epsilon exploration. High training reward alone does not prove a good frozen greedy policy; evaluation is separate. Alpha controls the update size, while epsilon controls random action selection.

Evidence: results/q_learning_results.json and results/q_learning_return.svg / q_learning_success.svg.

## 2. Part II - Single-robot DQN

### Task 6: Architecture and learning

State is a length-25 one-hot vector with index 5*row+column. Network is 25 -> 64 ReLU -> 64 ReLU -> 5 linear Q outputs; Q values are not probabilities and there is no softmax.

| Setting | Value |
|---|---:|
| Seed | 7, set before network construction |
| Environment steps | 12,000 |
| Initial random collection | 500 |
| Replay capacity / batch | 20,000 / 64 |
| Optimizer | Adam, learning rate 0.001 |
| Discount | 0.95 |
| Optimizer updates | Every 4 environment steps after warm-up |
| Target synchronization | Every 250 optimizer updates |
| Epsilon | 1.0 to 0.05 over first 80% of steps |
| Loss | Mean squared TD error |

The target is y=r+gamma*(1-d)*max Q_target(s',a'), where d=1 only for true termination. Truncation stores d=0 and retains the pre-reset next state. Target values are calculated under torch.no_grad(). gather() selects Q(s,a) for the action actually recorded in replay.

Hand checks:
| Case | r | max next Q | d | Target |
|---|---:|---:|---:|---:|
| Non-terminal | -1 | 2 | 0 | 0.9 |
| True terminal | +5 | ignored | 1 | 5.0 |
| Truncated | -1 | 2 | 0 | 0.9 |

The initial code seeded Torch after constructing networks and synchronized targets using environment-step arithmetic. That synced after 125 optimizer updates at the first attempted sync. The corrected code seeds before model construction and uses an explicit optimizer-update counter.

Measured results:
- 12,000 environment steps, seed 7; 1,455 completed episodes.
- Training success 94.36%, timeout 5.64%, mean invalid moves 0.819.
- Final 50-episode success 100%.
- 2,875 optimizer updates; 11 target syncs.
- Frozen greedy default start: success, 4 steps, zero invalid moves.
- Additional start (0,0): success, 6 steps, zero invalid moves.

The route matches the tabular shortest route. This tiny environment does not establish that DQN is more sample-efficient or preferable to tabular Q-learning.

Evidence: results/single_robot_dqn_results.json and the single_dqn SVG plots.

## 3. Part III - Two-robot DQN

### Task 7: Joint state/action and movement

Robot A starts (2,0), goal (2,4); B starts (2,4), goal (2,0). State is the ordered pair (pA,pB), represented by concatenating two 25-element one-hot vectors (50 inputs). There are 25 joint actions, encoded j=5*aA+aB and decoded by divmod(j,5). Network: 50 -> 64 ReLU -> 64 ReLU -> 25 linear outputs.

Both proposed movements are computed from the same old state. Robots at their own goal remain there. Same-cell and swap collisions terminate with reward -10 and retain both old positions. Entering a waiting robot cell is a collision. Following a robot into a cell it vacates is allowed if final cells differ and there is no swap.

For non-collision transitions, r=-0.1-b+5*k+5*I, where b is invalid moves by robots not already at goal, k is newly reached goals, and I=1 only if both are now at their goals. Collision overrides other reward terms. Terminate on collision or both goals; truncate unfinished episodes after 40 joint steps.

Required movement tests:
| Old A,B | A action | B action | Result |
|---|---|---|---|
| (2,1),(2,3) | Right | Left | Both propose (2,2): same-cell collision |
| (2,1),(2,2) | Right | Left | Swap collision |
| (2,1),(2,2) | Right | Right | B vacates (2,2), A follows: allowed |
| (2,1),(2,2) | Up | Wait | Allowed, positions (1,1),(2,2) |
| (2,1),(2,2) | Right | Wait | A enters waiting B's cell: collision |

Tests also verify action encoding, goal occupancy/no repeated arrival reward, collision priority, and output dimensions.

Manual collision-free route (not the learned result):
- A top route: (2,0)->(1,0)->(0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4).
- B bottom route: (2,4)->(3,4)->(4,4)->(4,3)->(4,2)->(4,1)->(4,0)->(3,0)->(2,0).

### Measured run and learned evaluation

Seed 7, 60,000 joint environment steps, same replay/network/optimizer/target settings as Task 6. PyTorch uses one CPU thread for these small networks. Run completed 14,875 optimizer updates and 59 target synchronizations.

| Training metric | Result |
|---|---:|
| Completed episodes | 2,166 |
| Success rate | 28.90% |
| Collision rate | 28.53% |
| Timeout rate | 42.57% |
| Mean invalid moves / episode | 5.57 |
| Final 50-episode success | 96% |

The training run contains exploratory failures. Frozen greedy evaluation:
- Default start: success in 9 joint steps, no collision/timeout/invalid moves. B enters doorway (2,2) first at step 2, reaches its goal (2,0) at step 4 and stays. A later crosses and reaches (2,4) at step 9.
- Additional start A=(0,0), B=(4,4), original goals unchanged: success in 10 joint steps, no collision/timeout/invalid moves.

These are actual network rollouts, not manual paths. Full paths are stored in results/two_robot_dqn_results.json and the joint path plot in results/two_robot_greedy_trajectory.svg.

## 4. Task 8 - Comparison and reflection

| Controller | Budget / seed | Default success | Collision | Timeout | Steps | Invalid |
|---|---|---|---|---|---:|---:|
| Tabular Q-learning | 1,500 episodes / 7 | Yes | N/A | No | 4 | 0 |
| One-robot DQN | 12,000 steps / 7 | Yes | N/A | No | 4 | 0 |
| Two-robot DQN | 60,000 joint steps / 7 | Yes | No | No | 9 joint | 0 |

Additional-start evaluation without retraining: tabular Q succeeds from (0,0) in 6 steps; single DQN succeeds from (0,0) in 6; two-robot DQN succeeds from A=(0,0),B=(4,4) in 10 joint steps. This is not exhaustive state-space generalization. Raw returns should not be compared across single-/two-robot tasks because reward definitions differ.

Reflection:
1. Design fixed before implementation: map, actions, rewards, terminal/truncated conditions, update, encodings, networks, replay settings, target sync, joint action encoding, simultaneous movement, and collision rules.
2. Initial code did not match every requirement: Torch seeded after network creation; target sync based on environment steps rather than optimizer updates; wrong shortest-path and reward-sign statements; missing evaluations. Corrections are in the code/report and covered by tests.
3. Tests exposed errors in doorway/path, negative reward TD arithmetic, terminal-vs-truncation masks, same-cell vs swap, waiting-cell collision, and following into vacated cells.
4. Replay and target network are present: bounded replay, random batches, copied target net, detached TD targets, sync every 250 optimizer updates.
5. Explainable functions: one-hot, joint encoding, simultaneous step, epsilon-greedy, replay sampling, gather, TD target, MSE, optimizer, sync, greedy evaluation.
6. Evidence: seed moved before model construction; explicit optimizer counter; only true terminal masks bootstrap; all required collision cases tested; greedy trajectories saved.

Optional RLHF: People could compare trajectory pairs and prefer safer/smoother outcomes. A reward model trained on these preferences predicts a scalar score; the policy can optimize it subject to safety constraints. Editing a hand-written collision penalty is not reward-model learning; it directly changes the specified reward.

## 5. Verification

Command: python -m unittest -v test_rl_lab.py  
Result: 18 tests passed, 0 failed.

Coverage includes movement/reward/termination, truncation, TD update arithmetic, epsilon schedule, joint-action encoding, all sheet collision cases, goal occupancy, DQN target values, and network dimensions.

Run:
python -m pip install -r requirements.txt
python -m unittest -v test_rl_lab.py
python tabular_q_learning_implementation.py
python dqn_single_robot.py
python dqn_two_robots.py
