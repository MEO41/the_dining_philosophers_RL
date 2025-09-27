import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import random
from collections import deque


class DQNReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """DQN Network for multi-agent discrete actions"""
    
    def __init__(self, state_dim: int, action_dim: int, n_philosophers: int, hidden_dim: int = 256):
        super().__init__()
        
        self.n_philosophers = n_philosophers
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate Q-networks for each philosopher
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(n_philosophers)
        ])
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all philosophers"""
        shared_features = self.shared_net(state)
        
        # Get Q-values for each philosopher
        q_values = []
        for i in range(self.n_philosophers):
            q_vals = self.q_heads[i](shared_features)
            q_values.append(q_vals)
        
        # Stack to get shape [batch_size, n_philosophers, action_dim]
        return torch.stack(q_values, dim=1)


class DQNAgent:
    """DQN Agent for discrete multi-agent actions"""
    
    def __init__(self, state_dim: int, action_dim: int, n_philosophers: int, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_philosophers = n_philosophers
        self.action_dim = action_dim
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, n_philosophers, config['hidden_dim']).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, n_philosophers, config['hidden_dim']).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        
        # Hyperparameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        
        # Replay buffer
        self.replay_buffer = DQNReplayBuffer(config['buffer_size'])
        
        # Training counters
        self.total_steps = 0
        self.update_count = 0
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if not deterministic and random.random() < self.epsilon:
            # Random action for each philosopher
            actions = np.random.randint(0, self.action_dim, size=self.n_philosophers)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                actions = q_values.argmax(dim=2).cpu().numpy()[0]
        
        return actions
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update(self) -> Dict[str, float]:
        """Update Q-network using DQN algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        
        # Gather Q-values for taken actions
        current_q = torch.zeros(self.batch_size, self.n_philosophers).to(self.device)
        for i in range(self.n_philosophers):
            current_q[:, i] = current_q_values[:, i, :].gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q = next_q_values.max(dim=2)[0]  # Max over actions for each philosopher
            
            # Calculate target for each philosopher
            target_q = torch.zeros(self.batch_size, self.n_philosophers).to(self.device)
            for i in range(self.n_philosophers):
                target_q[:, i] = rewards + (1 - dones) * self.gamma * next_q[:, i]
        
        # Calculate loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_values_mean': current_q.mean().item()
        }
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']