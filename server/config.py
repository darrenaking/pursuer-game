"""
Hyperparameters and configuration for the Pursuer training system.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Neural network architecture parameters."""
    input_size: int = 13       # State dimension
    hidden_size: int = 64      # GRU hidden dimension
    dense_size: int = 64       # Dense layer size
    action_size: int = 2       # [turn_rate, acceleration]
    
    # Action bounds (applied after tanh)
    max_turn_rate: float = 4.0      # radians per second
    max_acceleration: float = 800.0  # pixels per second squared


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE parameter
    clip_epsilon: float = 0.2        # PPO clipping
    entropy_coef: float = 0.01       # Entropy bonus for exploration
    value_coef: float = 0.5          # Value loss weight
    max_grad_norm: float = 0.5       # Gradient clipping
    
    # Batching
    batch_size: int = 32             # Trajectories per batch
    minibatch_size: int = 256        # Samples per minibatch
    epochs_per_update: int = 4       # PPO epochs per batch
    
    # Buffer
    buffer_size: int = 100           # Max trajectories in buffer
    min_trajectories: int = 16       # Min trajectories before training


@dataclass
class GameConfig:
    """Game parameters (must match frontend)."""
    episode_duration: float = 10.0   # seconds
    catch_radius: float = 30.0       # pixels
    
    # Pursuer constraints
    max_speed: float = 350.0         # pixels per second
    max_turn_rate: float = 4.0       # radians per second
    max_acceleration: float = 800.0  # pixels per second squared
    
    # Field of view
    fov_angle: float = 1.884         # radians (~108 degrees)
    fov_range: float = 400.0         # pixels
    
    # Arena (default, actual size comes from client)
    default_width: float = 1920.0
    default_height: float = 1080.0


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    weights_path: str = "weights/latest.json"
    save_interval: int = 10          # Save weights every N updates
    log_interval: int = 1            # Log stats every N updates


# Global config instances
model_config = ModelConfig()
training_config = TrainingConfig()
game_config = GameConfig()
server_config = ServerConfig()
