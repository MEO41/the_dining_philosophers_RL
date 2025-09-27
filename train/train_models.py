#!/usr/bin/env python3
"""
Training script for dining philosophers RL models
Usage: python train_models.py <algorithm>
Where algorithm is 'ppo', 'a2c', 'sac', or 'dqn'
"""

import sys
import os
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rl_models'))

from discrete_dining_philosophers_env import DiscreteDiningPhilosophersEnv
from continuous_dining_philosophers_env import ContinuousDiningPhilosophersEnv
from ppo import PPOAgent
from a2c import A2CAgent
from sac import SACAgent
from dqn import DQNAgent
from configs import get_config, get_env_config, EXPERIMENT_CONFIG


class Trainer:
    """Training manager for RL agents"""
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm.lower()
        self.config = get_config(self.algorithm)
        self.env_config = get_env_config(self.config['env_type'])
        
        self.setup_directories()
        self.env = self.create_environment()
        self.agent = self.create_agent()
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eating_counts = []
        self.fairness_scores = []
        self.losses = {}
        
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
    def setup_directories(self):
        """Create necessary directories"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.algorithm}_{self.timestamp}"
        
        for dir_name in ['save_dir', 'log_dir', 'results_dir']:
            dir_path = Path(EXPERIMENT_CONFIG[dir_name]) / self.run_name
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)
    
    def create_environment(self):
        """Create environment based on algorithm"""
        if self.config['env_type'] == 'discrete':
            return DiscreteDiningPhilosophersEnv(
                n_philosophers=self.config['n_philosophers'],
                max_steps=self.config['max_steps']
            )
        else:
            return ContinuousDiningPhilosophersEnv(
                n_philosophers=self.config['n_philosophers'],
                max_steps=self.config['max_steps']
            )
    
    def create_agent(self):
        """Create RL agent based on algorithm"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env_config['action_dim']
        
        agents = {
            'ppo': lambda: PPOAgent(state_dim, action_dim * self.config['n_philosophers'], self.config),
            'a2c': lambda: A2CAgent(state_dim, action_dim, self.config['n_philosophers'], self.config),
            'sac': lambda: SACAgent(state_dim, action_dim * self.config['n_philosophers'], self.config),
            'dqn': lambda: DQNAgent(state_dim, action_dim, self.config['n_philosophers'], self.config)
        }
        
        return agents[self.algorithm]()
    
    def reshape_action(self, action: np.ndarray) -> np.ndarray:
        """Reshape action for environment"""
        if self.algorithm in ['ppo', 'sac']:
            return action.reshape(self.config['n_philosophers'], 4)
        return action
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = self.agent.select_action(state)
            action_reshaped = self.reshape_action(action)
            next_state, reward, done, info = self.env.step(action_reshaped)
            
            # Store transition based on algorithm type
            if self.algorithm in ['sac', 'dqn']:
                self.agent.store_transition(state, action, reward, next_state, done)
            else:
                self.agent.store_transition(reward, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'eating_counts': info['eating_counts'],
            'info': info
        }
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        eval_rewards = []
        eval_eating_counts = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                if self.algorithm in ['sac', 'dqn']:
                    action = self.agent.select_action(state, deterministic=True)
                else:
                    action = self.agent.select_action(state)
                
                action_reshaped = self.reshape_action(action)
                state, reward, done, info = self.env.step(action_reshaped)
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_eating_counts.append(info['eating_counts'])
        
        fairness_scores = [1.0 / (1.0 + np.var(counts)) for counts in eval_eating_counts]
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_eating_events': np.mean([np.sum(counts) for counts in eval_eating_counts]),
            'mean_fairness': np.mean(fairness_scores),
            'eating_counts': eval_eating_counts
        }
    
    def update_agent(self, episode: int) -> Dict[str, float]:
        """Update agent if needed"""
        if self.algorithm in ['sac', 'dqn']:
            return self.agent.update()
        elif episode % self.config['update_frequency'] == 0:
            return self.agent.update()
        return {}
    
    def store_losses(self, losses: Dict[str, float]):
        """Store loss values based on algorithm"""
        loss_mappings = {
            'sac': {'actor_loss': 'actor', 'critic1_loss': 'critic1', 'critic2_loss': 'critic2', 'alpha': 'alpha'},
            'dqn': {'loss': 'loss', 'epsilon': 'epsilon', 'q_values_mean': 'q_values'},
            'ppo': {'actor_loss': 'actor', 'critic_loss': 'critic', 'entropy': 'entropy'},
            'a2c': {'actor_loss': 'actor', 'critic_loss': 'critic', 'entropy': 'entropy'}
        }
        
        mapping = loss_mappings[self.algorithm]
        for loss_key, store_key in mapping.items():
            if loss_key in losses:
                if store_key not in self.losses:
                    self.losses[store_key] = []
                self.losses[store_key].append(losses[loss_key])
    
    def log_progress(self, episode: int, episode_data: Dict[str, Any], losses: Dict[str, float]):
        """Log training progress"""
        if episode % self.config['log_frequency'] == 0:
            recent_rewards = self.episode_rewards[-self.config['log_frequency']:]
            recent_lengths = self.episode_lengths[-self.config['log_frequency']:]
            
            print(f"Episode {episode}")
            print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
            print(f"  Avg Length: {np.mean(recent_lengths):.1f}")
            print(f"  Eating Events: {np.sum(episode_data['eating_counts'])}")
            
            if losses:
                if self.algorithm == 'sac':
                    print(f"  Actor Loss: {losses.get('actor_loss', 0):.4f}")
                    print(f"  Alpha: {losses.get('alpha', 0):.4f}")
                elif self.algorithm == 'dqn':
                    print(f"  Loss: {losses.get('loss', 0):.4f}")
                    print(f"  Epsilon: {losses.get('epsilon', 0):.3f}")
                else:
                    print(f"  Actor Loss: {losses.get('actor_loss', 0):.4f}")
                    print(f"  Critic Loss: {losses.get('critic_loss', 0):.4f}")
            print("-" * 50)
    
    def save_model(self, episode: int, is_best: bool = False):
        """Save model checkpoint"""
        if episode % self.config['save_frequency'] == 0 or is_best:
            suffix = "_best" if is_best else f"_ep{episode}"
            model_path = self.save_dir / f"{self.algorithm}_model{suffix}.pth"
            self.agent.save(str(model_path))
            if is_best:
                print(f"Best model saved: {model_path}")
    
    def save_results(self):
        """Save training results"""
        results = {
            'algorithm': self.algorithm,
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eating_counts': self.eating_counts,
            'fairness_scores': self.fairness_scores,
            'losses': self.losses,
            'best_reward': self.best_reward
        }
        
        results_path = self.results_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Fairness scores
        if self.fairness_scores:
            axes[1, 0].plot(self.fairness_scores)
            axes[1, 0].set_title('Fairness Scores')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Fairness')
        
        # Primary loss
        primary_loss_key = {'sac': 'actor', 'dqn': 'loss', 'ppo': 'actor', 'a2c': 'actor'}[self.algorithm]
        if primary_loss_key in self.losses:
            axes[1, 1].plot(self.losses[primary_loss_key])
            axes[1, 1].set_title(f'{self.algorithm.upper()} Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png')
        plt.close()
    
    def check_early_stopping(self, current_reward: float) -> bool:
        """Check if training should stop early"""
        if not EXPERIMENT_CONFIG['early_stopping']['enabled']:
            return False
        
        if current_reward > self.best_reward + EXPERIMENT_CONFIG['early_stopping']['min_delta']:
            self.best_reward = current_reward
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            return self.no_improvement_count >= EXPERIMENT_CONFIG['early_stopping']['patience']
    
    def train(self):
        """Main training loop"""
        print(f"Starting {self.algorithm.upper()} training...")
        print(f"Environment: {self.config['env_type']}")
        print(f"Philosophers: {self.config['n_philosophers']}")
        print(f"Episodes: {self.config['n_episodes']}")
        print("=" * 60)
        
        for episode in range(1, self.config['n_episodes'] + 1):
            # Train episode
            episode_data = self.train_episode()
            
            # Store metrics
            self.episode_rewards.append(episode_data['episode_reward'])
            self.episode_lengths.append(episode_data['episode_length'])
            self.eating_counts.append(episode_data['eating_counts'])
            
            # Calculate fairness
            eating_variance = np.var(episode_data['eating_counts'])
            fairness = 1.0 / (1.0 + eating_variance)
            self.fairness_scores.append(fairness)
            
            # Update agent
            losses = self.update_agent(episode)
            if losses:
                self.store_losses(losses)
            
            # Log progress
            self.log_progress(episode, episode_data, losses)
            
            # Evaluation
            if episode % self.config['eval_frequency'] == 0:
                eval_results = self.evaluate(self.config['eval_episodes'])
                print(f"\nEvaluation at episode {episode}:")
                print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"  Mean Eating Events: {eval_results['mean_eating_events']:.1f}")
                print(f"  Mean Fairness: {eval_results['mean_fairness']:.3f}")
                
                # Save best model
                if eval_results['mean_reward'] > self.best_reward:
                    self.best_reward = eval_results['mean_reward']
                    self.save_model(episode, is_best=True)
                
                # Early stopping
                if self.check_early_stopping(eval_results['mean_reward']):
                    print(f"\nEarly stopping at episode {episode}")
                    break
            
            # Save checkpoint
            self.save_model(episode)
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_eval = self.evaluate(50)
        print(f"Final Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"Final Mean Eating Events: {final_eval['mean_eating_events']:.1f}")
        print(f"Final Mean Fairness: {final_eval['mean_fairness']:.3f}")
        
        # Save results
        self.save_results()
        self.plot_training_curves()
        
        print(f"\nTraining completed! Best reward: {self.best_reward:.2f}")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python train_models.py <algorithm>")
        print("algorithm should be 'ppo', 'a2c', 'sac', or 'dqn'")
        sys.exit(1)
    
    algorithm = sys.argv[1].lower()
    
    if algorithm not in ['ppo', 'a2c', 'sac', 'dqn']:
        print(f"Unknown algorithm: {algorithm}")
        print("Use 'ppo', 'a2c', 'sac', or 'dqn'")
        sys.exit(1)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        trainer = Trainer(algorithm)
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()