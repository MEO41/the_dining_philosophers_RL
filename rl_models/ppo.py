import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import gymnasium as gym


class PPOActor(nn.Module):
    """PPO Actor network for continuous actions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        self.action_dim = action_dim
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action means and log stds"""
        output = self.net(state)
        
        mean = output[:, :self.action_dim]
        log_std = output[:, self.action_dim:]
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return action, log_prob, entropy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        
        # Apply sigmoid to ensure actions are in [0, 1]
        action = torch.sigmoid(action)
        
        # Calculate log probability
        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob, entropy
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action given state"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Inverse sigmoid to get original action before sigmoid
        action_raw = torch.log(action / (1 - action + 1e-8))
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action_raw)
        
        return log_prob.sum(dim=-1, keepdim=True)


class PPOCritic(nn.Module):
    """PPO Critic network"""
    
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


class PPOAgent:
    """PPO Agent for continuous action spaces"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = PPOActor(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.critic = PPOCritic(state_dim, config['hidden_dim']).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        # Hyperparameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.c1 = config['value_coef']
        self.c2 = config['entropy_coef']
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
            action, log_prob, _ = self.actor.get_action(state_tensor)
            value = self.critic(state_tensor)
        
        # Store for training
        self.states.append(state)
        self.actions.append(action.cpu().numpy()[0])
        self.log_probs.append(log_prob.cpu().numpy()[0])
        self.values.append(value.cpu().numpy()[0])
        
        return action.cpu().numpy()[0]
    
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
        log_probs = self.log_probs[:min_length]
        rewards = self.rewards[:min_length]
        values = self.values[:min_length]
        dones = self.dones[:min_length]
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        rewards = np.array(rewards)
        values = np.array(values).flatten()
        dones = np.array(dones)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, values, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = returns - torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # PPO epochs
            # Get current policy predictions
            new_log_probs = self.actor.get_log_prob(states, actions)
            values_pred = self.critic(states).squeeze()
            
            # Calculate ratios
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate actor loss
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.unsqueeze(1)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate critic loss
            critic_loss = F.mse_loss(values_pred, returns)
            
            # Calculate entropy
            _, _, entropy = self.actor.get_action(states)
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        # Clear storage
        self.clear_memory()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'returns_mean': returns.mean().item()
        }
    
    def _calculate_returns(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Calculate discounted returns using GAE"""
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # lambda = 0.95
            returns[t] = gae + values[t]
        
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