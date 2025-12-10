"""
Neural network architecture for the Pursuer agent.

Actor-Critic with GRU for memory:
    Input → Dense → GRU → Dense → Policy Head (actor)
                              → Value Head (critic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import json
import os

from config import model_config


class PursuerNetwork(nn.Module):
    """
    GRU-based actor-critic network for continuous control.
    
    Outputs:
        - action_mean: Mean of Gaussian policy for each action dimension
        - action_std: Standard deviation (learned parameter)
        - value: State value estimate
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        cfg = config or model_config
        
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.dense_size = cfg.dense_size
        self.action_size = cfg.action_size
        
        # Input projection
        self.input_dense = nn.Linear(cfg.input_size, cfg.dense_size)
        
        # Recurrent layer
        self.gru = nn.GRU(
            input_size=cfg.dense_size,
            hidden_size=cfg.hidden_size,
            batch_first=True
        )
        
        # Post-GRU processing
        self.post_gru_dense = nn.Linear(cfg.hidden_size, cfg.dense_size)
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(cfg.dense_size, cfg.action_size)
        
        # Learnable log standard deviation
        self.policy_log_std = nn.Parameter(torch.zeros(cfg.action_size))
        
        # Value head (critic)
        self.value_head = nn.Linear(cfg.dense_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Smaller init for policy output (encourages exploration initially)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        
        # Value head init
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or (batch, input_size)
            hidden: GRU hidden state of shape (1, batch, hidden_size) or None
            
        Returns:
            action_mean: Shape (batch, [seq_len,] action_size)
            action_std: Shape (action_size,)
            value: Shape (batch, [seq_len,] 1)
            new_hidden: Shape (1, batch, hidden_size)
        """
        # Handle single timestep input
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        
        # Input projection
        x = F.relu(self.input_dense(x))
        
        # GRU
        gru_out, new_hidden = self.gru(x, hidden)
        
        # Post-GRU dense
        x = F.relu(self.post_gru_dense(gru_out))
        
        # Policy head
        action_mean = torch.tanh(self.policy_mean(x))  # Bounded to [-1, 1]
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        # Value head
        value = self.value_head(x)
        
        # Remove seq dimension if single step
        if single_step:
            action_mean = action_mean.squeeze(1)
            value = value.squeeze(1)
        
        return action_mean, action_std, value, new_hidden
    
    def get_action(self, x, hidden=None, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            x: State tensor
            hidden: GRU hidden state
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Sampled or mean action
            log_prob: Log probability of the action
            value: State value estimate
            new_hidden: Updated hidden state
        """
        action_mean, action_std, value, new_hidden = self.forward(x, hidden)
        
        if deterministic:
            return action_mean, None, value, new_hidden
        
        # Create distribution and sample
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        # Clamp to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, new_hidden
    
    def evaluate_actions(self, x, actions, hidden=None):
        """
        Evaluate log probability and entropy of given actions.
        
        Used during PPO update to compute policy loss.
        
        Args:
            x: State tensor (batch, seq_len, input_size)
            actions: Action tensor (batch, seq_len, action_size)
            hidden: Initial hidden state
            
        Returns:
            log_probs: Log probability of each action
            entropy: Policy entropy
            values: State value estimates
        """
        action_mean, action_std, values, _ = self.forward(x, hidden)
        
        dist = Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy, values
    
    def get_initial_hidden(self, batch_size=1, device='cpu'):
        """Return zeroed initial hidden state."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
    def export_weights(self):
        """
        Export weights to JSON-serializable format for JavaScript inference.
        
        Returns:
            dict: Weights organized for JS GRU implementation
        """
        state = self.state_dict()
        
        weights = {
            'input_dense': {
                'weight': state['input_dense.weight'].tolist(),
                'bias': state['input_dense.bias'].tolist()
            },
            'gru': {
                # GRU has weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
                'weight_ih': state['gru.weight_ih_l0'].tolist(),
                'weight_hh': state['gru.weight_hh_l0'].tolist(),
                'bias_ih': state['gru.bias_ih_l0'].tolist(),
                'bias_hh': state['gru.bias_hh_l0'].tolist()
            },
            'post_gru_dense': {
                'weight': state['post_gru_dense.weight'].tolist(),
                'bias': state['post_gru_dense.bias'].tolist()
            },
            'policy_mean': {
                'weight': state['policy_mean.weight'].tolist(),
                'bias': state['policy_mean.bias'].tolist()
            },
            'policy_log_std': state['policy_log_std'].tolist(),
            'value_head': {
                'weight': state['value_head.weight'].tolist(),
                'bias': state['value_head.bias'].tolist()
            },
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'dense_size': self.dense_size,
                'action_size': self.action_size
            }
        }
        
        return weights
    
    def save_weights(self, path):
        """Save weights to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        weights = self.export_weights()
        with open(path, 'w') as f:
            json.dump(weights, f)
    
    def load_weights_from_json(self, path):
        """Load weights from JSON file (for continuity from JS-trained model)."""
        with open(path, 'r') as f:
            weights = json.load(f)
        
        state = self.state_dict()
        
        state['input_dense.weight'] = torch.tensor(weights['input_dense']['weight'])
        state['input_dense.bias'] = torch.tensor(weights['input_dense']['bias'])
        
        state['gru.weight_ih_l0'] = torch.tensor(weights['gru']['weight_ih'])
        state['gru.weight_hh_l0'] = torch.tensor(weights['gru']['weight_hh'])
        state['gru.bias_ih_l0'] = torch.tensor(weights['gru']['bias_ih'])
        state['gru.bias_hh_l0'] = torch.tensor(weights['gru']['bias_hh'])
        
        state['post_gru_dense.weight'] = torch.tensor(weights['post_gru_dense']['weight'])
        state['post_gru_dense.bias'] = torch.tensor(weights['post_gru_dense']['bias'])
        
        state['policy_mean.weight'] = torch.tensor(weights['policy_mean']['weight'])
        state['policy_mean.bias'] = torch.tensor(weights['policy_mean']['bias'])
        
        state['policy_log_std'] = torch.tensor(weights['policy_log_std'])
        
        state['value_head.weight'] = torch.tensor(weights['value_head']['weight'])
        state['value_head.bias'] = torch.tensor(weights['value_head']['bias'])
        
        self.load_state_dict(state)


if __name__ == '__main__':
    # Test the network
    net = PursuerNetwork()
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, model_config.input_size)
    
    action_mean, action_std, value, hidden = net(x)
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Action std shape: {action_std.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test single step
    x_single = torch.randn(batch_size, model_config.input_size)
    action, log_prob, value, hidden = net.get_action(x_single)
    print(f"\nSingle step action shape: {action.shape}")
    
    # Test export
    weights = net.export_weights()
    print(f"\nExported weight keys: {list(weights.keys())}")
