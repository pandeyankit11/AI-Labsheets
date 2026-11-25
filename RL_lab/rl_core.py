"""Verified tabular Q-learning and DQN implementations for the CS-F407 RL lab."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable
import numpy as np
import torch
from torch import nn

GRID_SIZE = 5
OBSTACLES = frozenset({(0, 2), (1, 2), (3, 2), (4, 2)})
START = (2, 0)
GOAL = (2, 4)
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ACTION_NAMES = ("Up", "Down", "Left", "Right", "Wait")
GAMMA = 0.95

def set_all_seeds(seed: int) -> None:
    # Small CPU batches are faster and more repeatable without large thread pools.
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def valid_position(pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and pos not in OBSTACLES

def position_index(pos: tuple[int, int]) -> int:
    return pos[0] * GRID_SIZE + pos[1]

def one_hot_position(pos: tuple[int, int]) -> np.ndarray:
    vec = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    vec[position_index(pos)] = 1.0
    return vec

class WarehouseEnv:
    """Single robot; centre cell (2,2) is the open doorway."""
    def __init__(self, max_steps: int = 40):
        self.max_steps = max_steps
        self.pos = START
        self.steps = 0
    def reset(self, start: tuple[int, int] = START) -> tuple[int, int]:
        if not valid_position(start):
            raise ValueError(f"Invalid reset position: {start}")
        self.pos = tuple(start)
        self.steps = 0
        return self.pos
    def set_position(self, pos: tuple[int, int]) -> None:
        if not valid_position(pos):
            raise ValueError(f"Invalid position: {pos}")
        self.pos = tuple(pos)
        self.steps = 0
    def step(self, action: int):
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(ACTIONS):
            raise ValueError("Action must be an integer in [0, 4]")
        if self.pos == GOAL:
            raise RuntimeError("Cannot step an episode after true termination")
        dr, dc = ACTIONS[int(action)]
        candidate = (self.pos[0] + dr, self.pos[1] + dc)
        invalid = not valid_position(candidate)
        next_pos = self.pos if invalid else candidate
        reward = -1.0 if invalid else -0.1
        terminated = next_pos == GOAL
        if terminated:
            reward = 10.0
        self.pos = next_pos
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        return next_pos, reward, terminated, truncated, {"invalid_move": invalid}

class TwoRobotWarehouseEnv:
    """Simultaneous two-robot environment with same-cell and swap collisions."""
    def __init__(self, max_steps: int = 40):
        self.max_steps = max_steps
        self.start_A, self.start_B = (2, 0), (2, 4)
        self.goal_A, self.goal_B = (2, 4), (2, 0)
        self.pos_A, self.pos_B = self.start_A, self.start_B
        self.steps = 0
    def reset(self, start_A=None, start_B=None):
        self.pos_A = self.start_A if start_A is None else tuple(start_A)
        self.pos_B = self.start_B if start_B is None else tuple(start_B)
        if not valid_position(self.pos_A) or not valid_position(self.pos_B):
            raise ValueError("Robot start positions must be traversable grid cells")
        if self.pos_A == self.pos_B:
            raise ValueError("Robots cannot start in the same cell")
        self.steps = 0
        return self.pos_A, self.pos_B
    def _propose(self, pos, action, goal):
        if pos == goal:
            return pos, True
        dr, dc = ACTIONS[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not valid_position(nxt):
            return pos, False
        return nxt, True
    @staticmethod
    def encode_joint_action(action_A: int, action_B: int) -> int:
        if not (0 <= action_A < 5 and 0 <= action_B < 5):
            raise ValueError("Each robot action must be in [0, 4]")
        return 5 * action_A + action_B
    @staticmethod
    def decode_joint_action(joint_action: int) -> tuple[int, int]:
        if not isinstance(joint_action, (int, np.integer)) or not 0 <= int(joint_action) < 25:
            raise ValueError("Joint action must be an integer in [0, 24]")
        return divmod(int(joint_action), 5)
    def step(self, joint_action: int):
        a_A, a_B = self.decode_joint_action(joint_action)
        old_A, old_B = self.pos_A, self.pos_B
        prop_A, valid_A = self._propose(old_A, a_A, self.goal_A)
        prop_B, valid_B = self._propose(old_B, a_B, self.goal_B)
        same_cell = prop_A == prop_B
        swap = prop_A == old_B and prop_B == old_A and old_A != old_B
        collision = same_cell or swap
        self.steps += 1
        if collision:
            reward = -10.0
            terminated = True
            next_A, next_B = old_A, old_B
            invalid_count = int(not valid_A and old_A != self.goal_A) + int(not valid_B and old_B != self.goal_B)
            newly_reached = 0
        else:
            invalid_count = int(not valid_A and old_A != self.goal_A) + int(not valid_B and old_B != self.goal_B)
            newly_reached = int(old_A != self.goal_A and prop_A == self.goal_A) + int(old_B != self.goal_B and prop_B == self.goal_B)
            next_A, next_B = prop_A, prop_B
            both_at_goals = next_A == self.goal_A and next_B == self.goal_B
            reward = -0.1 - invalid_count + 5.0 * newly_reached + (5.0 if both_at_goals else 0.0)
            terminated = both_at_goals
        self.pos_A, self.pos_B = next_A, next_B
        truncated = self.steps >= self.max_steps and not terminated
        info = {"collision": collision, "same_cell_collision": same_cell,
                "swap_collision": swap, "invalid_moves": invalid_count,
                "newly_reached_goals": newly_reached}
        return (next_A, next_B), float(reward), terminated, truncated, info

def q_learning_target(reward: float, next_max_q: float, gamma: float, terminated: bool) -> float:
    return float(reward if terminated else reward + gamma * next_max_q)

def q_learning_update(old_q: float, reward: float, next_max_q: float, alpha: float,
                      gamma: float, terminated: bool) -> tuple[float, float]:
    target = q_learning_target(reward, next_max_q, gamma, terminated)
    return float(old_q + alpha * (target - old_q)), target

def epsilon_for_episode(episode: int, decay_episodes: int = 1200) -> float:
    if episode < 0:
        raise ValueError("episode must be non-negative")
    fraction = min(episode / max(decay_episodes - 1, 1), 1.0)
    return 1.0 - 0.95 * fraction

def train_q_learning(seed: int = 7, episodes: int = 1500):
    rng = np.random.default_rng(seed)
    env = WarehouseEnv()
    Q = np.zeros((25, 5), dtype=np.float64)
    alpha, gamma = 0.2, GAMMA
    returns, successes, timeouts, invalid_counts, epsilons = [], [], [], [], []
    for episode in range(episodes):
        pos = env.reset()
        ep_return = 0.0
        invalid = 0
        epsilon = epsilon_for_episode(episode)
        epsilons.append(epsilon)
        terminated = truncated = False
        while not (terminated or truncated):
            s = position_index(pos)
            if rng.random() < epsilon:
                action = int(rng.integers(5))
            else:
                ties = np.flatnonzero(Q[s] == Q[s].max())
                action = int(rng.choice(ties))
            next_pos, reward, terminated, truncated, info = env.step(action)
            invalid += int(info["invalid_move"])
            ep_return += reward
            ns = position_index(next_pos)
            Q[s, action], _ = q_learning_update(Q[s, action], reward, float(Q[ns].max()), alpha, gamma, terminated)
            pos = next_pos
        returns.append(float(ep_return))
        successes.append(int(terminated))
        timeouts.append(int(truncated))
        invalid_counts.append(invalid)
    return Q, {"returns": returns, "successes": successes, "timeouts": timeouts,
               "invalid_moves": invalid_counts, "epsilons": epsilons, "seed": seed,
               "episodes": episodes, "alpha": alpha, "gamma": gamma}

def evaluate_q_policy(Q: np.ndarray, start=START, max_steps: int = 40):
    env = WarehouseEnv(max_steps=max_steps)
    pos = env.reset(start)
    path, actions = [pos], []
    invalid = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(np.argmax(Q[position_index(pos)]))
        actions.append(action)
        pos, _, terminated, truncated, info = env.step(action)
        invalid += int(info["invalid_move"])
        path.append(pos)
    return {"success": bool(terminated), "timeout": bool(truncated), "steps": len(actions),
            "invalid_moves": invalid, "path": path, "actions": actions}

def evaluate_random_policy(episodes: int = 100, seed: int = 17):
    rng = np.random.default_rng(seed)
    successes, timeouts, lengths, invalid = [], [], [], []
    for _ in range(episodes):
        env = WarehouseEnv()
        env.reset()
        term = trunc = False
        n_invalid = steps = 0
        while not (term or trunc):
            _, _, term, trunc, info = env.step(int(rng.integers(5)))
            n_invalid += int(info["invalid_move"])
            steps += 1
        successes.append(int(term)); timeouts.append(int(trunc))
        lengths.append(steps); invalid.append(n_invalid)
    successful_lengths = [n for n, ok in zip(lengths, successes) if ok]
    return {"episodes": episodes, "seed": seed, "success_rate": float(np.mean(successes)),
            "timeout_rate": float(np.mean(timeouts)),
            "mean_steps_successful": float(np.mean(successful_lengths)) if successful_lengths else None,
            "mean_invalid_moves": float(np.mean(invalid))}

class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, output_size))
    def forward(self, x):
        return self.net(x)

@dataclass
class DQNConfig:
    total_steps: int
    seed: int = 7
    warmup_steps: int = 500
    replay_capacity: int = 20_000
    batch_size: int = 64
    update_every: int = 4
    target_sync_updates: int = 250
    learning_rate: float = 0.001
    gamma: float = GAMMA
    epsilon_end_fraction: float = 0.8
    epsilon_final: float = 0.05

def dqn_target(reward: torch.Tensor, next_q: torch.Tensor, terminated: torch.Tensor,
               gamma: float = GAMMA) -> torch.Tensor:
    return reward + gamma * (1.0 - terminated) * next_q

def state_features_single(pos):
    return one_hot_position(pos)

def state_features_multi(state):
    return np.concatenate((one_hot_position(state[0]), one_hot_position(state[1])))

def epsilon_for_step(step: int, config: DQNConfig) -> float:
    decay_steps = max(1, int(config.total_steps * config.epsilon_end_fraction))
    frac = min(max(step, 0) / decay_steps, 1.0)
    return max(config.epsilon_final, 1.0 - (1.0 - config.epsilon_final) * frac)

def evaluate_network_policy(network, feature_fn, env, start, step_fn):
    if isinstance(env, WarehouseEnv):
        state = env.reset(start)
    else:
        state = env.reset(start[0], start[1])
    path = [state]
    invalid = collisions = steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            x = torch.as_tensor(feature_fn(state), dtype=torch.float32).unsqueeze(0)
            action = int(network(x).argmax(dim=1).item())
        state, _, terminated, truncated, info = step_fn(env, action)
        path.append(state)
        invalid += int(info.get("invalid_moves", int(info.get("invalid_move", False))))
        collisions += int(info.get("collision", False))
        steps += 1
    return {"success": bool(terminated and collisions == 0), "collision": bool(collisions),
            "timeout": bool(truncated), "steps": steps, "invalid_moves": invalid, "path": path}

def _run_dqn(env, config: DQNConfig, input_size: int, output_size: int,
             feature_fn, reset_fn, step_fn, evaluation_starts: Iterable):
    set_all_seeds(config.seed)  # seed before network construction
    online, target = QNetwork(input_size, output_size), QNetwork(input_size, output_size)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = deque(maxlen=config.replay_capacity)
    state = reset_fn(env)
    episode_return = 0.0
    episode_invalid = 0
    episode_returns, episode_successes, episode_timeouts, episode_invalids = [], [], [], []
    optimizer_updates = 0
    loss_values = []
    for step in range(1, config.total_steps + 1):
        eps = epsilon_for_step(step, config)
        if step <= config.warmup_steps or random.random() < eps:
            action = random.randrange(output_size)
        else:
            with torch.no_grad():
                x = torch.as_tensor(feature_fn(state), dtype=torch.float32).unsqueeze(0)
                action = int(online(x).argmax(dim=1).item())
        next_state, reward, terminated, truncated, info = step_fn(env, action)
        replay.append((state, action, float(reward), next_state, bool(terminated)))
        episode_return += float(reward)
        episode_invalid += int(info.get("invalid_moves", int(info.get("invalid_move", False))))
        if terminated or truncated:
            episode_returns.append(float(episode_return))
            episode_successes.append(int(terminated and not info.get("collision", False)))
            episode_timeouts.append(int(truncated))
            episode_invalids.append(episode_invalid)
            state = reset_fn(env)
            episode_return, episode_invalid = 0.0, 0
        else:
            state = next_state
        if step > config.warmup_steps and step % config.update_every == 0 and len(replay) >= config.batch_size:
            batch = random.sample(replay, config.batch_size)
            states, actions, rewards, next_states, terminals = zip(*batch)
            states_t = torch.as_tensor(np.asarray([feature_fn(s) for s in states]), dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
            next_states_t = torch.as_tensor(np.asarray([feature_fn(s) for s in next_states]), dtype=torch.float32)
            terminals_t = torch.as_tensor(terminals, dtype=torch.float32)
            chosen_q = online(states_t).gather(1, actions_t).squeeze(1)
            with torch.no_grad():
                next_q = target(next_states_t).max(dim=1).values
                targets = dqn_target(rewards_t, next_q, terminals_t, config.gamma)
            loss = nn.functional.mse_loss(chosen_q, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            loss_values.append(float(loss.item()))
            if optimizer_updates % config.target_sync_updates == 0:
                target.load_state_dict(online.state_dict())
    results = {"config": config.__dict__, "optimizer_updates": optimizer_updates,
               "target_sync_updates_requested": optimizer_updates // config.target_sync_updates,
               "loss_last": loss_values[-1] if loss_values else None,
               "episode_returns": episode_returns, "episode_successes": episode_successes,
               "episode_timeouts": episode_timeouts, "episode_invalid_moves": episode_invalids}
    results["evaluation"] = {label: evaluate_network_policy(online, feature_fn, env, start, step_fn)
                             for label, start in evaluation_starts}
    return online, results

def _single_reset(env): return env.reset()
def _single_step(env, action): return env.step(action)
def _multi_reset(env): return env.reset()
def _multi_step(env, action): return env.step(action)

def train_single_dqn(seed: int = 7, total_steps: int = 12_000):
    config = DQNConfig(total_steps=total_steps, seed=seed)
    env = WarehouseEnv()
    return _run_dqn(env, config, 25, 5, state_features_single, _single_reset, _single_step,
                    [("default_start", START), ("additional_start", (0, 0))])

def train_two_robot_dqn(seed: int = 7, total_steps: int = 60_000):
    config = DQNConfig(total_steps=total_steps, seed=seed)
    env = TwoRobotWarehouseEnv()
    return _run_dqn(env, config, 50, 25, state_features_multi, _multi_reset, _multi_step,
                    [("default_start", ((2, 0), (2, 4))),
                     ("additional_start", ((0, 0), (4, 4)))])
