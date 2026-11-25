import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Reuse environment logic from Part 1
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
        return self.pos

    def step(self, action):
        r, c = self.pos
        new_r, new_c = r, c
        if action == 0: new_r -= 1
        elif action == 1: new_r += 1
        elif action == 2: new_c -= 1
        elif action == 3: new_c += 1
        
        reward = -0.1
        terminated = False
        
        if (new_r < 0 or new_r >= self.grid_size or 
            new_c < 0 or new_c >= self.grid_size or 
            (new_r, new_c) in self.obstacles):
            self.pos = (r, c)
            reward = -1.0
        else:
            self.pos = (new_r, new_c)
            
        if self.pos == self.goal:
            reward = 10.0
            terminated = True
            
        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        return self.pos, reward, terminated, truncated

def pos_to_one_hot(pos):
    idx = pos[0] * 5 + pos[1]
    vec = np.zeros(25, dtype=np.float32)
    vec[idx] = 1.0
    return vec

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
    def forward(self, x):
        return self.net(x)

def train_dqn():
    env = WarehouseEnv()
    
    online_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    replay_buffer = deque(maxlen=20000)
    batch_size = 64
    gamma = 0.95
    
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    
    total_steps = 12000
    collect_steps = 500
    
    state = env.reset()
    
    for step in range(1, total_steps + 1):
        # Epsilon decay over first 80% of training (12000 * 0.8 = 9600)
        epsilon = max(0.05, 1.0 - (1.0 - 0.05) * (step / 9600))
        
        if step <= collect_steps or random.random() < epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                q_vals = online_net(torch.tensor(pos_to_one_hot(state)))
                action = q_vals.argmax().item()
                
        next_state, reward, terminated, truncated = env.step(action)
        
        replay_buffer.append((state, action, reward, next_state, terminated))
        
        if terminated or truncated:
            state = env.reset()
        else:
            state = next_state
            
        # Training
        if step > collect_steps and step % 4 == 0:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, terms = zip(*batch)
            
            states_t = torch.tensor(np.array([pos_to_one_hot(s) for s in states]))
            actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            next_states_t = torch.tensor(np.array([pos_to_one_hot(s) for s in next_states]))
            terms_t = torch.tensor(terms, dtype=torch.float32)
            
            q_values = online_net(states_t).gather(1, actions_t).squeeze()
            
            with torch.no_grad():
                next_q_values = target_net(next_states_t).max(1)[0]
                targets = rewards_t + gamma * (1 - terms_t) * next_q_values
                
            loss = nn.MSELoss()(q_values, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step // 4) % 250 == 0:
                target_net.load_state_dict(online_net.state_dict())

if __name__ == '__main__':
    train_dqn()
    print("DQN Training Complete.")