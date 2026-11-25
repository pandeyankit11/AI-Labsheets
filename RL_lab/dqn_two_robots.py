import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class TwoRobotWarehouseEnv:
    def __init__(self):
        self.grid_size = 5
        self.obstacles = {(0, 2), (1, 2), (3, 2), (4, 2)}
        self.start_A = (2, 0)
        self.start_B = (2, 4)
        self.goal_A = (2, 4)
        self.goal_B = (2, 0)
        self.max_steps = 40
        self.reset()

    def reset(self):
        self.pos_A = self.start_A
        self.pos_B = self.start_B
        self.steps = 0
        return self.pos_A, self.pos_B

    def _proposed_move(self, pos, action):
        r, c = pos
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1
        
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size or (r, c) in self.obstacles:
            return pos, False # Invalid move
        return (r, c), True

    def step(self, joint_action):
        action_A, action_B = divmod(joint_action, 5)
        
        # Robots already at goal stay at goal
        if self.pos_A == self.goal_A: action_A = 4
        if self.pos_B == self.goal_B: action_B = 4
        
        prop_A, valid_A = self._proposed_move(self.pos_A, action_A)
        prop_B, valid_B = self._proposed_move(self.pos_B, action_B)
        
        # Check Collisions
        collision = False
        if prop_A == prop_B:
            collision = True
        if prop_A == self.pos_B and prop_B == self.pos_A:
            collision = True
            
        reward = 0.0
        terminated = False
        
        if collision:
            reward = -10.0
            terminated = True
            # Retain old positions
        else:
            b = (0 if valid_A or self.pos_A == self.goal_A else 1) + \
                (0 if valid_B or self.pos_B == self.goal_B else 1)
            
            k = 0
            if prop_A == self.goal_A and self.pos_A != self.goal_A: k += 1
            if prop_B == self.goal_B and self.pos_B != self.goal_B: k += 1
            
            self.pos_A = prop_A
            self.pos_B = prop_B
            
            I = 1 if (self.pos_A == self.goal_A and self.pos_B == self.goal_B) else 0
            
            reward = -0.1 - b + 5*k + 5*I
            if I == 1:
                terminated = True
                
        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        return (self.pos_A, self.pos_B), reward, terminated, truncated

def state_to_input(pos_A, pos_B):
    vec = np.zeros(50, dtype=np.float32)
    vec[pos_A[0]*5 + pos_A[1]] = 1.0
    vec[25 + pos_B[0]*5 + pos_B[1]] = 1.0
    return vec

class MultiDQN(nn.Module):
    def __init__(self):
        super(MultiDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 25)
        )
        
    def forward(self, x):
        return self.net(x)

def train_multi_dqn():
    env = TwoRobotWarehouseEnv()
    online_net = MultiDQN()
    target_net = MultiDQN()
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)
    replay_buffer = deque(maxlen=20000)
    batch_size = 64
    gamma = 0.95
    
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    
    total_steps = 60000
    collect_steps = 500
    
    state = env.reset()
    
    for step in range(1, total_steps + 1):
        epsilon = max(0.05, 1.0 - (1.0 - 0.05) * (step / (total_steps * 0.8)))
        
        if step <= collect_steps or random.random() < epsilon:
            joint_action = random.randint(0, 24)
        else:
            with torch.no_grad():
                q_vals = online_net(torch.tensor(state_to_input(*state)))
                joint_action = q_vals.argmax().item()
                
        next_state, reward, terminated, truncated = env.step(joint_action)
        replay_buffer.append((state, joint_action, reward, next_state, terminated))
        
        if terminated or truncated:
            state = env.reset()
        else:
            state = next_state
            
        if step > collect_steps and step % 4 == 0:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, terms = zip(*batch)
            
            states_t = torch.tensor(np.array([state_to_input(*s) for s in states]))
            actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            next_states_t = torch.tensor(np.array([state_to_input(*s) for s in next_states]))
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
    train_multi_dqn()
    print("Multi-Agent DQN Training Complete.")