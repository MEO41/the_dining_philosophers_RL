import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class A2CActor(nn.Module):
    """A2C Actor network for discrete actions"""
    
    def __init__(self, state_dim: int, action_dim: int, n_philosophers: int, hidden_dim: int = 256):
        super().__init__()
        
        self.n_philosophers = n_philosophers
        self.action_dim = action_dim  # 6 actions per philosopher
        
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate action heads for each philosopher
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(n_philosophers)
        ])
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits for each philosopher"""
        shared_features = self.shared_net(state)
        
        # Get action logits for each philosopher
        action_logits = []
        for i in range(self.n_philosophers):
            logits = self.action_heads[i](shared_features)
            action_logits.append(logits)
        
        # Stack to get shape [batch_size, n_philosophers, action_dim]
        return torch.stack(action_logits, dim=1)
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and return actions, log_probs, entropy"""
        logits = self.forward(state)
        
        # Create categorical distributions for each philosopher
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Sample actions
        actions = dist.sample()
        
        # Calculate log probabilities
        log_probs = dist.log_prob(actions)
        
        # Calculate entropy
        entropy = dist.entropy()
        
        return actions, log_probs, entropy
    
    def get_log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions given state"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        return log_probs


class A2CCritic(nn.Module):
    """A2C Critic network"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        return self.net(state)


class A2CAgent:
    """A2C Agent for discrete action spaces"""
    
    def __init__(self, state_dim: int, action_dim: int, n_philosophers: int, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_philosophers = n_philosophers
        
        # Networks
        self.actor = A2CActor(state_dim, action_dim, n_philosophers, config['hidden_dim']).to(self.device)
        self.critic = A2CCritic(state_dim, config['hidden_dim']).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        # Hyperparameters
        self.gamma = config['gamma']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.max_grad_norm = config['max_grad_norm']
        
        # Storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actions, log_probs, _ = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        # Store for training
        self.states.append(state)
        self.actions.append(actions.cpu().numpy()[0])
        self.log_probs.append(log_probs.cpu().numpy()[0])
        self.values.append(value.cpu().numpy()[0])
        
        return actions.cpu().numpy()[0]
    
    def store_transition(self, reward: float, done: bool):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self) -> Dict[str, float]:
        """Update actor and critic networks"""
        if len(self.states) == 0:
            return {}
        
        # Ensure all lists have same length
        min_length = min(len(self.states), len(self.actions), len(self.log_probs), 
                        len(self.rewards), len(self.values), len(self.dones))
        
        # Truncate to minimum length
        states = self.states[:min_length]
        actions = self.actions[:min_length]
        rewards = self.rewards[:min_length]
        values = self.values[:min_length]
        dones = self.dones[:min_length]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = np.array(rewards)
        values = np.array(values).flatten()
        dones = np.array(dones)
        
        # Calculate returns
        returns = self._calculate_returns(rewards, values, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Calculate advantages
        advantages = returns - torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current predictions
        action_logits = self.actor(states)
        values_pred = self.critic(states).squeeze()
        
        # Calculate actor loss
        log_probs = self.actor.get_log_prob(states, actions)
        actor_loss = -(log_probs * advantages.unsqueeze(1)).mean()
        
        # Calculate critic loss
        critic_loss = F.mse_loss(values_pred, returns)
        
        # Calculate entropy
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Clear storage
        self.clear_memory()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'returns_mean': returns.mean().item()
        }
    
    def _calculate_returns(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Calculate discounted returns"""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])