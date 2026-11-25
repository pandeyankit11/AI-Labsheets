# Reinforcement Learning and Deep RL Lab Report

## Part I: Design and Test a Q-learning Agent

### Task 1: Understand the MDP
**MDP Specification:**
*   **State:** The current position of the robot in the 5x5 grid, encoded as an index $(r \times 5 + c)$.
*   **Actions:** 0 (Up), 1 (Down), 2 (Left), 3 (Right), 4 (Wait).
*   **Transition rule:** Deterministic movement to adjacent cell unless obstructed by boundary or obstacle, in which case the robot stays in the same cell. 
*   **Reward:** +10 on goal arrival, -1 on invalid move, -0.1 for other valid moves (including waiting).
*   **Initial state:** (2,0).
*   **Terminal outcome:** Reaching the goal (2,4) acts as true termination. 

**Answers:**
**(a) What information is necessary to specify the state?**
Only the current row and column coordinates.
**(b) Why is the current position sufficient when the map and goal are fixed?**
Since the map layout, obstacles, and goal are static, the environment obeys the Markov property; future transitions and rewards only depend on the current position.
**(c) Is the transition rule deterministic?**
Yes. Taking a valid action always moves the robot exactly one unit in that direction. 
**(d) What is the shortest valid route from S to G?**
S(2,0) -> (3,0) -> (3,1) -> (4,1) -> (4,2) -> (4,3) -> (3,3) -> (2,3) -> G(2,4) (Length = 8 steps, passing below the obstacle). Passing above is also 8 steps.
**(e) Is finding that route by search the same process as learning action values from experience?**
No. Search (like BFS/A*) uses a known model of the environment to plan a path ahead of time. Learning from experience involves acting in the environment without prior map knowledge, observing rewards, and iteratively estimating the value of actions.

### Task 4: Test the Environment and the Q-learning Update
**Synthetic Update Tests:**
Given: $Q(s,a)=0.4$, $r=1$, $max_{s'}Q(s',a')=2$, $\alpha=0.2$, $\gamma=0.95$.
*   **Non-terminal target:** $y = r + \gamma \max Q = 1 + 0.95(2) = 2.9$. 
    Update: $Q = 0.4 + 0.2(2.9 - 0.4) = 0.9$.
*   **Terminal target ($r=5$):** $y = r = 5$. 
    Update: $Q = 0.4 + 0.2(5 - 0.4) = 1.32$.
*   **Truncated target:** Behaves identically to non-terminal, so $y = 2.9$. Update = $0.9$.
*Which case should ignore the next-state value?* Only the True Termination case ignores the next-state value. Truncation must bootstrap using the next-state value.

### Task 5: Train, Evaluate, and Investigate Exploration
**Answers:**
**(a) Why might the training curve remain noisy after a good route is learned?**
Because of $\epsilon$-greedy exploration, the agent continues to take random actions sporadically during training, preventing perfect runs every time. 
**(b) Does a high training return necessarily imply a good greedy policy?**
No. An agent might learn a sub-optimal policy that avoids heavy penalties but takes longer, yielding a lower but stable training return, while a high return run might just be lucky exploration.
**(c) What is the difference between the learning rate $\alpha$ and the exploration probability $\epsilon$?**
$\alpha$ controls how quickly new reward information overrides old Q-value estimates. $\epsilon$ controls how frequently the agent picks random actions to explore the environment versus exploiting its learned table.

## Part II: Replace the Table with a Neural Network

### Task 6: DQN Validate
**DQN Target Calculations Test:**
*   Non-terminal ($r=1, \max Q=2$): Target = $1 + 0.95 \times 2 = 2.9$.
*   Terminal ($r=5, \max Q=2$): Target = $5 + 0.95 \times 0 \times 2 = 5$.
*   Truncated ($r=1, \max Q=2$): Target = $1 + 0.95 \times 2 = 2.9$. 
*(In my implementation, `terms_t` acts as `d`. It is 1 for terminal, 0 for truncated/normal).*

## Part III: Two Robots and Collision Avoidance

### Task 7: Design and Test a Two-Robot Controller
**(a) Explain why observing only A's position is insufficient.**
Without B's position, the agent cannot learn collision avoidance because the collision state is entirely hidden from the observation.
**(b) Write down how you will represent both positions.**
Concatenate two one-hot vectors of length 25 into a single 50-dimensional input vector.
**(c) Calculate the number of joint actions.**
$5 \times 5 = 25$ joint actions.
**(d) Draw one possible collision-free route for both robots.**
Robot A goes above the obstacle, Robot B goes below the obstacle.
**(e) Explain where one robot might need to wait or move aside.**
If both attempt to pass through the door cell (2,2) simultaneously or one wants to use a narrow corridor the other is occupying, one must issue the Wait(4) action until the path is clear.

**Testing Environment Outcomes:**
*   `Old (2,1), (2,3)`, Action (right, left) -> Predict: Swap Collision. Both return to old positions.
*   `Old (2,1), (2,2)`, Action (right, left) -> Predict: Swap Collision. Both return to old positions.
*   `Old (2,1), (2,2)`, Action (up, wait) -> Predict: Valid. A goes to (1,1), B stays at (2,2). 

**(a) Which robot entered the passage first?**
*(Based on general DQN trained results)* Usually, the agent learns an asymmetric policy where Robot B waits at (2,3) while Robot A clears the (2,2) doorway.
**(c) Is avoiding collisions sufficient to count as success?**
No. They must also reach their goals. A policy that makes both robots wait forever successfully avoids collisions but fails the delivery task.
**(d) How would the joint-action output size change for three robots?**
$5 \times 5 \times 5 = 125$ joint actions.

## Part IV: Evaluation and Reflection

### Task 8: Evaluate the Agents and the LLM's Contribution
**Reflection Questions:**
1. **Which parts of the design did you specify before requesting code?**
I specified state encodings (flat index vs one-hot vectors), network architectures (hidden layer sizes), reward structures, bounds on truncation, and collision rules.
2. **Did the LLM change any rewards, stopping rules, or algorithm details?**
By using highly specific prompts, the LLM faithfully implemented my requested reward math and strictly followed the separation of truncation and termination.
3. **Which generated functions can you now explain without consulting the LLM?**
I can explain the DQN TD target generation logic `targets = rewards + gamma * (1 - terms) * next_q` and how `gather()` extracts the action values from the model. 

### Optional: Connect to RLHF
1. **What feedback people would provide:** 
Humans would label/rank pairs of recorded episodes, selecting the one that looked "smoother" or had less unnecessary waiting.
2. **What a learned reward model would predict:** 
It would predict a scalar score evaluating the aesthetic or strategic quality of a given trajectory based on those human preferences.
3. **How an RL agent could use that:** 
The agent would replace the hard-coded environmental reward function (or augment it) with the scalar output from the learned reward model during training.