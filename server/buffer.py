"""
Trajectory buffer for storing and batching gameplay data.

Each trajectory is one episode of gameplay containing:
    - states: (T, state_dim) observations at each timestep
    - actions: (T, action_dim) actions taken
    - rewards: (T,) rewards received
    - dones: (T,) episode termination flags
    - hidden_init: Initial GRU hidden state for the episode
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import threading

from config import training_config, model_config


@dataclass
class Trajectory:
    """A single episode trajectory."""
    states: np.ndarray       # (T, state_dim)
    actions: np.ndarray      # (T, action_dim)
    rewards: np.ndarray      # (T,)
    dones: np.ndarray        # (T,)
    hidden_init: np.ndarray  # (hidden_dim,) - initial hidden state
    
    # Metadata
    episode_length: int
    total_reward: float
    caught: bool
    time_to_catch: Optional[float]
    
    def __len__(self):
        return self.episode_length


class TrajectoryBuffer:
    """
    Thread-safe buffer for collecting trajectories from multiple players.
    """
    
    def __init__(self, max_size=None):
        self.max_size = max_size or training_config.buffer_size
        self.trajectories = deque(maxlen=self.max_size)
        self.lock = threading.Lock()
        
        # Statistics
        self.total_episodes = 0
        self.total_catches = 0
        self.total_rewards = 0.0
        self.best_time = float('inf')
        
    def add(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        with self.lock:
            self.trajectories.append(trajectory)
            
            # Update stats
            self.total_episodes += 1
            self.total_rewards += trajectory.total_reward
            
            if trajectory.caught:
                self.total_catches += 1
                if trajectory.time_to_catch and trajectory.time_to_catch < self.best_time:
                    self.best_time = trajectory.time_to_catch
    
    def add_from_dict(self, data: dict):
        """
        Add trajectory from JSON data received from client.
        
        Expected format:
        {
            'states': [[...], [...], ...],
            'actions': [[...], [...], ...],
            'rewards': [...],
            'caught': bool,
            'time_to_catch': float or null,
            'episode_length': int
        }
        """
        states = np.array(data['states'], dtype=np.float32)
        actions = np.array(data['actions'], dtype=np.float32)
        rewards = np.array(data['rewards'], dtype=np.float32)
        
        episode_length = len(rewards)
        dones = np.zeros(episode_length, dtype=np.float32)
        dones[-1] = 1.0  # Last step is terminal
        
        # Client doesn't send hidden state (always starts at zero)
        hidden_init = np.zeros(model_config.hidden_size, dtype=np.float32)
        
        trajectory = Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            hidden_init=hidden_init,
            episode_length=episode_length,
            total_reward=float(rewards.sum()),
            caught=data.get('caught', False),
            time_to_catch=data.get('time_to_catch')
        )
        
        self.add(trajectory)
        return trajectory
    
    def sample_batch(self, batch_size=None) -> List[Trajectory]:
        """
        Sample a batch of trajectories for training.
        
        Returns all available trajectories up to batch_size, then clears them.
        """
        batch_size = batch_size or training_config.batch_size
        
        with self.lock:
            batch = list(self.trajectories)[:batch_size]
            # Remove sampled trajectories
            for _ in range(len(batch)):
                if self.trajectories:
                    self.trajectories.popleft()
            return batch
    
    def ready_for_training(self) -> bool:
        """Check if we have enough trajectories for a training batch."""
        return len(self.trajectories) >= training_config.min_trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def get_stats(self) -> dict:
        """Get buffer and training statistics."""
        with self.lock:
            catch_rate = self.total_catches / max(1, self.total_episodes)
            avg_reward = self.total_rewards / max(1, self.total_episodes)
            
            return {
                'buffer_size': len(self.trajectories),
                'total_episodes': self.total_episodes,
                'total_catches': self.total_catches,
                'catch_rate': catch_rate,
                'average_reward': avg_reward,
                'best_time': self.best_time if self.best_time < float('inf') else None
            }


def trajectories_to_tensors(trajectories: List[Trajectory], device='cpu'):
    """
    Convert a batch of trajectories to padded tensors for training.
    
    Returns:
        states: (batch, max_len, state_dim)
        actions: (batch, max_len, action_dim)
        rewards: (batch, max_len)
        dones: (batch, max_len)
        masks: (batch, max_len) - 1 for valid timesteps, 0 for padding
        hidden_inits: (1, batch, hidden_dim)
    """
    batch_size = len(trajectories)
    max_len = max(len(t) for t in trajectories)
    
    state_dim = trajectories[0].states.shape[1]
    action_dim = trajectories[0].actions.shape[1]
    hidden_dim = trajectories[0].hidden_init.shape[0]
    
    # Initialize padded arrays
    states = np.zeros((batch_size, max_len, state_dim), dtype=np.float32)
    actions = np.zeros((batch_size, max_len, action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size, max_len), dtype=np.float32)
    dones = np.zeros((batch_size, max_len), dtype=np.float32)
    masks = np.zeros((batch_size, max_len), dtype=np.float32)
    hidden_inits = np.zeros((batch_size, hidden_dim), dtype=np.float32)
    
    for i, traj in enumerate(trajectories):
        length = len(traj)
        states[i, :length] = traj.states
        actions[i, :length] = traj.actions
        rewards[i, :length] = traj.rewards
        dones[i, :length] = traj.dones
        masks[i, :length] = 1.0
        hidden_inits[i] = traj.hidden_init
    
    # Convert to tensors
    return (
        torch.tensor(states, device=device),
        torch.tensor(actions, device=device),
        torch.tensor(rewards, device=device),
        torch.tensor(dones, device=device),
        torch.tensor(masks, device=device),
        torch.tensor(hidden_inits, device=device).unsqueeze(0)  # (1, batch, hidden)
    )


if __name__ == '__main__':
    # Test buffer
    buffer = TrajectoryBuffer(max_size=10)
    
    # Create fake trajectory
    for i in range(5):
        traj = Trajectory(
            states=np.random.randn(100, model_config.input_size).astype(np.float32),
            actions=np.random.randn(100, model_config.action_size).astype(np.float32),
            rewards=np.random.randn(100).astype(np.float32),
            dones=np.zeros(100, dtype=np.float32),
            hidden_init=np.zeros(model_config.hidden_size, dtype=np.float32),
            episode_length=100,
            total_reward=float(np.random.randn()),
            caught=i % 2 == 0,
            time_to_catch=5.0 if i % 2 == 0 else None
        )
        buffer.add(traj)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")
    print(f"Ready for training: {buffer.ready_for_training()}")
    
    # Test batch conversion
    batch = buffer.sample_batch(3)
    tensors = trajectories_to_tensors(batch)
    print(f"\nBatch tensors:")
    print(f"  States: {tensors[0].shape}")
    print(f"  Actions: {tensors[1].shape}")
    print(f"  Rewards: {tensors[2].shape}")
    print(f"  Masks: {tensors[4].shape}")
