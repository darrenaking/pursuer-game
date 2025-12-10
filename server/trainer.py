"""
PPO (Proximal Policy Optimization) trainer for the Pursuer agent.

Implements:
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus for exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional
import threading
import time

from model import PursuerNetwork
from buffer import TrajectoryBuffer, Trajectory, trajectories_to_tensors
from config import training_config, model_config, server_config


class PPOTrainer:
    """
    PPO trainer that runs in background and updates model from trajectory buffer.
    """
    
    def __init__(self, model: PursuerNetwork, buffer: TrajectoryBuffer, device='cpu'):
        self.model = model
        self.buffer = buffer
        self.device = device
        
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.learning_rate
        )
        
        # Training state
        self.update_count = 0
        self.running = False
        self.train_thread = None
        
        # Logging
        self.last_losses = {}
        self.training_stats = {
            'updates': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        masks: torch.Tensor
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: (batch, seq_len)
            values: (batch, seq_len)
            dones: (batch, seq_len)
            masks: (batch, seq_len) - valid timestep mask
            
        Returns:
            advantages: (batch, seq_len)
            returns: (batch, seq_len)
        """
        batch_size, seq_len = rewards.shape
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gamma = training_config.gamma
        gae_lambda = training_config.gae_lambda
        
        for b in range(batch_size):
            last_gae = 0
            last_value = 0
            
            # Work backwards through sequence
            for t in reversed(range(seq_len)):
                if masks[b, t] == 0:
                    continue
                    
                if t == seq_len - 1 or masks[b, t + 1] == 0:
                    # Last valid timestep
                    next_value = 0
                    next_non_terminal = 0
                else:
                    next_value = values[b, t + 1]
                    next_non_terminal = 1 - dones[b, t]
                
                delta = rewards[b, t] + gamma * next_value * next_non_terminal - values[b, t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                
                advantages[b, t] = last_gae
                returns[b, t] = advantages[b, t] + values[b, t]
        
        return advantages, returns
    
    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Perform PPO update on a batch of trajectories.
        
        Returns:
            Dictionary of loss values for logging
        """
        if not trajectories:
            return {}
        
        # Convert to tensors
        states, actions, rewards, dones, masks, hidden_inits = trajectories_to_tensors(
            trajectories, device=self.device
        )
        
        batch_size, seq_len, _ = states.shape
        
        # Get initial values and log probs (before update)
        with torch.no_grad():
            _, _, old_values, _ = self.model(states, hidden_inits)
            old_values = old_values.squeeze(-1)
            
            # Compute old log probs
            old_log_probs, _, _ = self.model.evaluate_actions(states, actions, hidden_inits)
            old_log_probs = old_log_probs.squeeze(-1)
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, old_values, dones, masks)
        
        # Normalize advantages
        valid_advs = advantages[masks.bool()]
        if len(valid_advs) > 1:
            advantages = (advantages - valid_advs.mean()) / (valid_advs.std() + 1e-8)
        
        # Flatten for minibatching
        flat_states = states.view(-1, states.shape[-1])
        flat_actions = actions.view(-1, actions.shape[-1])
        flat_old_log_probs = old_log_probs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        flat_masks = masks.view(-1)
        
        # Get valid indices only
        valid_indices = flat_masks.nonzero().squeeze(-1)
        
        if len(valid_indices) == 0:
            return {}
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(training_config.epochs_per_update):
            # Shuffle valid indices
            perm = torch.randperm(len(valid_indices))
            
            for start in range(0, len(valid_indices), training_config.minibatch_size):
                end = min(start + training_config.minibatch_size, len(valid_indices))
                mb_indices = valid_indices[perm[start:end]]
                
                # Get minibatch data
                mb_states = flat_states[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_old_log_probs = flat_old_log_probs[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]
                
                # Forward pass (no hidden state for flattened single steps)
                # This is a simplification - ideally we'd maintain hidden states
                action_mean, action_std, values, _ = self.model(mb_states)
                values = values.squeeze(-1)
                
                # Compute new log probs
                from torch.distributions import Normal
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - training_config.clip_epsilon,
                    1 + training_config.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)
                
                # Total loss
                loss = (
                    policy_loss
                    + training_config.value_coef * value_loss
                    - training_config.entropy_coef * entropy
                )
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    training_config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        self.update_count += 1
        
        # Average losses
        losses = {
            'policy_loss': total_policy_loss / max(1, num_updates),
            'value_loss': total_value_loss / max(1, num_updates),
            'entropy': total_entropy / max(1, num_updates),
            'num_trajectories': len(trajectories),
            'total_timesteps': int(masks.sum().item())
        }
        
        self.last_losses = losses
        self.training_stats['updates'] = self.update_count
        self.training_stats['policy_loss'] = losses['policy_loss']
        self.training_stats['value_loss'] = losses['value_loss']
        self.training_stats['entropy'] = losses['entropy']
        
        return losses
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Check buffer and run training if enough data available.
        
        Returns:
            Loss dict if training happened, None otherwise
        """
        if not self.buffer.ready_for_training():
            return None
        
        # Get batch
        trajectories = self.buffer.sample_batch()
        
        if not trajectories:
            return None
        
        # Train
        losses = self.update(trajectories)
        
        # Save weights periodically
        if self.update_count % server_config.save_interval == 0:
            self.save_weights()
        
        # Log periodically
        if self.update_count % server_config.log_interval == 0:
            print(f"Update {self.update_count}: "
                  f"policy_loss={losses['policy_loss']:.4f}, "
                  f"value_loss={losses['value_loss']:.4f}, "
                  f"entropy={losses['entropy']:.4f}, "
                  f"trajectories={losses['num_trajectories']}")
        
        return losses
    
    def training_loop(self, interval=1.0):
        """
        Background training loop.
        
        Args:
            interval: Seconds between training checks
        """
        print("Training loop started")
        
        while self.running:
            self.train_step()
            time.sleep(interval)
        
        print("Training loop stopped")
    
    def start(self, interval=1.0):
        """Start background training thread."""
        if self.running:
            return
        
        self.running = True
        self.train_thread = threading.Thread(
            target=self.training_loop,
            args=(interval,),
            daemon=True
        )
        self.train_thread.start()
    
    def stop(self):
        """Stop background training thread."""
        self.running = False
        if self.train_thread:
            self.train_thread.join(timeout=5.0)
    
    def save_weights(self, path=None):
        """Save model weights to JSON."""
        path = path or server_config.weights_path
        self.model.save_weights(path)
        print(f"Saved weights to {path}")
    
    def load_weights(self, path=None):
        """Load model weights from JSON."""
        path = path or server_config.weights_path
        try:
            self.model.load_weights_from_json(path)
            print(f"Loaded weights from {path}")
        except FileNotFoundError:
            print(f"No existing weights at {path}, starting fresh")
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            **self.training_stats,
            **self.buffer.get_stats()
        }


if __name__ == '__main__':
    # Test trainer
    model = PursuerNetwork()
    buffer = TrajectoryBuffer()
    trainer = PPOTrainer(model, buffer)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Add some fake trajectories
    for i in range(20):
        traj = Trajectory(
            states=np.random.randn(50, model_config.input_size).astype(np.float32),
            actions=np.random.randn(50, model_config.action_size).astype(np.float32),
            rewards=np.random.randn(50).astype(np.float32),
            dones=np.zeros(50, dtype=np.float32),
            hidden_init=np.zeros(model_config.hidden_size, dtype=np.float32),
            episode_length=50,
            total_reward=float(np.random.randn()),
            caught=i % 2 == 0,
            time_to_catch=5.0 if i % 2 == 0 else None
        )
        traj.dones[-1] = 1.0
        buffer.add(traj)
    
    print(f"\nBuffer has {len(buffer)} trajectories")
    print(f"Ready for training: {buffer.ready_for_training()}")
    
    # Run a training step
    losses = trainer.train_step()
    print(f"\nTraining losses: {losses}")
    
    # Test weight export
    weights = model.export_weights()
    print(f"\nWeight shapes:")
    print(f"  GRU weight_ih: {len(weights['gru']['weight_ih'])}x{len(weights['gru']['weight_ih'][0])}")
    print(f"  GRU weight_hh: {len(weights['gru']['weight_hh'])}x{len(weights['gru']['weight_hh'][0])}")
