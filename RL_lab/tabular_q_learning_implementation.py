import numpy as np
import matplotlib.pyplot as plt

# --- Environment Definition ---
class WarehouseEnv:
    def __init__(self):
        self.grid_size = 5
        self.obstacles = {(0, 2), (1, 2), (3, 2), (4, 2)}
        self.start = (2, 0)
        self.goal = (2, 4)
        self.max_steps = 40
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.state_to_idx(self.pos)

    def state_to_idx(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def step(self, action):
        r, c = self.pos
        new_r, new_c = r, c
        
        if action == 0: new_r -= 1       # Up
        elif action == 1: new_r += 1     # Down
        elif action == 2: new_c -= 1     # Left
        elif action == 3: new_c += 1     # Right
        # action == 4 is Wait (stays same)

        reward = -0.1
        terminated = False
        
        # Check boundary and obstacles
        if (new_r < 0 or new_r >= self.grid_size or 
            new_c < 0 or new_c >= self.grid_size or 
            (new_r, new_c) in self.obstacles):
            self.pos = (r, c) # Stay in place
            reward = -1.0
        else:
            self.pos = (new_r, new_c)
            
        if self.pos == self.goal:
            reward = 10.0
            terminated = True
            
        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        return self.state_to_idx(self.pos), reward, terminated, truncated

# --- Agent and Training ---
def train_q_learning():
    env = WarehouseEnv()
    n_states = 25
    n_actions = 5
    
    Q = np.zeros((n_states, n_actions))
    alpha = 0.2
    gamma = 0.95
    n_episodes = 1500
    
    np.random.seed(7)
    
    returns = []
    successes = []
    
    for ep in range(n_episodes):
        state = env.reset()
        ep_return = 0
        
        # Linear epsilon decay from 1.0 to 0.05 over 1200 episodes
        epsilon = max(0.05, 1.0 - (1.0 - 0.05) * (ep / 1200))
        
        while True:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                # Random tie-breaking
                max_val = np.max(Q[state])
                action = np.random.choice(np.flatnonzero(Q[state] == max_val))
                
            next_state, reward, terminated, truncated = env.step(action)
            ep_return += reward
            
            # Q-learning target computation
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
                
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
            
            if terminated or truncated:
                returns.append(ep_return)
                successes.append(1 if terminated else 0)
                break
                
    return Q, returns, successes

# --- Evaluation ---
def evaluate_greedy(Q):
    env = WarehouseEnv()
    state = env.reset()
    path = [env.pos]
    
    while True:
        # Fixed tie-breaking for evaluation (e.g., first max action)
        action = np.argmax(Q[state])
        state, reward, terminated, truncated = env.step(action)
        path.append(env.pos)
        
        if terminated or truncated:
            break
            
    print("Greedy Trajectory:", path)
    print("Length:", len(path)-1)
    
if __name__ == '__main__':
    Q, returns, successes = train_q_learning()
    evaluate_greedy(Q)