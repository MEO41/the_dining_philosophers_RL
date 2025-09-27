#!/usr/bin/env python3
"""
Trained RL Models Evaluation
Comprehensive evaluation of trained RL agents with detailed analysis
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rl_models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from discrete_dining_philosophers_env import DiscreteDiningPhilosophersEnv
from continuous_dining_philosophers_env import ContinuousDiningPhilosophersEnv


class TrainedRLEvaluator:
    """Comprehensive evaluator for trained RL models"""
    
    def __init__(self, models_dir: str = "models", n_philosophers: int = 5):
        self.models_dir = Path(models_dir)
        self.n_philosophers = n_philosophers
        self.results_dir = Path("evaluate/rl_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation metrics
        self.metrics = [
            'eating_events', 'fairness', 'episode_length', 'reward',
            'eating_variance', 'deadlock_events', 'efficiency'
        ]
    
    def find_trained_models(self) -> Dict[str, Path]:
        """Find all trained models"""
        models = {}
        algorithms = ['ppo', 'a2c', 'sac', 'dqn']
        
        for algorithm in algorithms:
            # Find directories containing the algorithm name
            algo_dirs = [d for d in self.models_dir.iterdir() 
                        if d.is_dir() and algorithm in d.name.lower()]
            
            if algo_dirs:
                # Use most recent directory
                latest_dir = max(algo_dirs, key=lambda x: x.stat().st_mtime)
                best_model_path = latest_dir / f"{algorithm}_model_best.pth"
                
                if best_model_path.exists():
                    models[algorithm] = best_model_path
                    print(f"Found {algorithm} model: {best_model_path}")
        
        return models
    
    def load_agent_and_env(self, algorithm: str, model_path: Path) -> Tuple[Any, Any]:
        """Load trained agent and corresponding environment"""
        if algorithm == 'ppo':
            from ppo import PPOAgent
            from configs import PPO_CONFIG
            env = ContinuousDiningPhilosophersEnv(n_philosophers=self.n_philosophers)
            state_dim = env.observation_space.shape[0]
            action_dim = 4 * self.n_philosophers
            agent = PPOAgent(state_dim, action_dim, PPO_CONFIG)
            
        elif algorithm == 'a2c':
            from a2c import A2CAgent
            from configs import A2C_CONFIG
            env = DiscreteDiningPhilosophersEnv(n_philosophers=self.n_philosophers)
            state_dim = env.observation_space.shape[0]
            agent = A2CAgent(state_dim, 6, self.n_philosophers, A2C_CONFIG)
            
        elif algorithm == 'sac':
            from sac import SACAgent
            from configs import SAC_CONFIG
            env = ContinuousDiningPhilosophersEnv(n_philosophers=self.n_philosophers)
            state_dim = env.observation_space.shape[0]
            action_dim = 4 * self.n_philosophers
            agent = SACAgent(state_dim, action_dim, SAC_CONFIG)
            
        elif algorithm == 'dqn':
            from dqn import DQNAgent
            from configs import DQN_CONFIG
            env = DiscreteDiningPhilosophersEnv(n_philosophers=self.n_philosophers)
            state_dim = env.observation_space.shape[0]
            agent = DQNAgent(state_dim, 6, self.n_philosophers, DQN_CONFIG)
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        agent.load(str(model_path))
        return agent, env
    
    def evaluate_single_episode(self, agent: Any, env: Any, algorithm: str, 
                               max_steps: int = 1000, render: bool = False) -> Dict[str, Any]:
        """Evaluate single episode"""
        state = env.reset()
        episode_reward = 0
        steps = 0
        deadlock_steps = 0
        
        if render:
            env.render()
        
        while steps < max_steps:
            # Get action from agent
            if algorithm in ['sac', 'dqn']:
                action = agent.select_action(state, deterministic=True)
            else:
                action = agent.select_action(state)
            
            # Reshape action for environment
            if algorithm in ['ppo', 'sac']:
                action_reshaped = action.reshape(self.n_philosophers, 4)
            else:
                action_reshaped = action
            
            state, reward, done, info = env.step(action_reshaped)
            episode_reward += reward
            steps += 1
            
            # Count deadlock steps if applicable
            if 'deadlock_steps' in info:
                deadlock_steps += info['deadlock_steps']
            
            if render and steps % 50 == 0:
                env.render()
            
            if done:
                break
        
        # Calculate metrics
        eating_counts = info['eating_counts']
        total_eating_events = np.sum(eating_counts)
        eating_variance = np.var(eating_counts)
        fairness = 1.0 / (1.0 + eating_variance)
        efficiency = total_eating_events / steps if steps > 0 else 0
        
        return {
            'eating_counts': eating_counts.tolist(),
            'total_eating_events': total_eating_events,
            'eating_variance': eating_variance,
            'fairness': fairness,
            'episode_reward': episode_reward,
            'episode_length': steps,
            'deadlock_steps': deadlock_steps,
            'efficiency': efficiency,
            'info': info
        }
    
    def evaluate_algorithm(self, algorithm: str, model_path: Path, 
                          n_episodes: int = 50, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate algorithm over multiple episodes"""
        print(f"Evaluating {algorithm.upper()}...")
        
        try:
            agent, env = self.load_agent_and_env(algorithm, model_path)
        except Exception as e:
            print(f"Error loading {algorithm}: {e}")
            return {'algorithm': algorithm, 'error': str(e)}
        
        results = []
        start_time = time.time()
        
        for episode in range(n_episodes):
            if episode % 10 == 0:
                print(f"  Episode {episode}/{n_episodes}")
            
            result = self.evaluate_single_episode(agent, env, algorithm, max_steps)
            results.append(result)
        
        duration = time.time() - start_time
        print(f"  Completed in {duration:.2f}s")
        
        # Aggregate results
        metrics = {
            'algorithm': algorithm,
            'n_episodes': n_episodes,
            'evaluation_duration': duration,
            'model_path': str(model_path)
        }
        
        # Calculate statistics for each metric
        for metric in ['total_eating_events', 'fairness', 'episode_reward', 
                      'episode_length', 'eating_variance', 'deadlock_steps', 'efficiency']:
            values = [r[metric] for r in results]
            metrics[f'mean_{metric}'] = np.mean(values)
            metrics[f'std_{metric}'] = np.std(values)
            metrics[f'min_{metric}'] = np.min(values)
            metrics[f'max_{metric}'] = np.max(values)
            metrics[f'median_{metric}'] = np.median(values)
        
        # Eating distribution analysis
        all_eating_counts = [r['eating_counts'] for r in results]
        metrics['eating_distribution'] = {
            'mean_per_philosopher': np.mean(all_eating_counts, axis=0).tolist(),
            'std_per_philosopher': np.std(all_eating_counts, axis=0).tolist()
        }
        
        # Success rate (episodes with eating events)
        successful_episodes = sum(1 for r in results if r['total_eating_events'] > 0)
        metrics['success_rate'] = successful_episodes / n_episodes
        
        metrics['detailed_results'] = results
        
        return metrics
    
    def evaluate_all(self, n_episodes: int = 50, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate all available trained models"""
        models = self.find_trained_models()
        
        if not models:
            print("No trained models found!")
            return {}
        
        print(f"Found {len(models)} trained models")
        print("Starting evaluation...")
        
        all_results = {}
        
        for algorithm, model_path in models.items():
            all_results[algorithm] = self.evaluate_algorithm(
                algorithm, model_path, n_episodes, max_steps
            )
        
        return all_results
    
    def create_performance_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create performance comparison dataframe"""
        data = []
        
        for algorithm, result in results.items():
            if 'error' not in result:
                data.append({
                    'Algorithm': algorithm.upper(),
                    'Mean_Eating_Events': result['mean_total_eating_events'],
                    'Std_Eating_Events': result['std_total_eating_events'],
                    'Mean_Fairness': result['mean_fairness'],
                    'Std_Fairness': result['std_fairness'],
                    'Mean_Reward': result['mean_episode_reward'],
                    'Std_Reward': result['std_episode_reward'],
                    'Mean_Length': result['mean_episode_length'],
                    'Success_Rate': result['success_rate'],
                    'Efficiency': result['mean_efficiency']
                })
        
        return pd.DataFrame(data)
    
    def plot_performance_comparison(self, df: pd.DataFrame):
        """Create comprehensive performance plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = [
            ('Mean_Eating_Events', 'Average Eating Events per Episode'),
            ('Mean_Fairness', 'Fairness Index'),
            ('Mean_Reward', 'Average Episode Reward'),
            ('Success_Rate', 'Success Rate'),
            ('Efficiency', 'Eating Efficiency (events/step)'),
            ('Mean_Length', 'Average Episode Length')
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (metric, title) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            bars = ax.bar(df['Algorithm'], df[metric], color=colors[:len(df)], alpha=0.7)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' '))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'rl_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_eating_distribution(self, results: Dict[str, Any]):
        """Plot eating distribution across philosophers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (algorithm, result) in enumerate(results.items()):
            if 'error' in result or idx >= 4:
                continue
                
            ax = axes[idx]
            
            distribution = result['eating_distribution']
            philosopher_ids = list(range(self.n_philosophers))
            means = distribution['mean_per_philosopher']
            stds = distribution['std_per_philosopher']
            
            bars = ax.bar(philosopher_ids, means, yerr=stds, 
                         capsize=5, alpha=0.7, color=f'C{idx}')
            ax.set_title(f'{algorithm.upper()} - Eating Distribution')
            ax.set_xlabel('Philosopher ID')
            ax.set_ylabel('Average Eating Count')
            ax.set_xticks(philosopher_ids)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.1, f'{mean:.1f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'eating_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, results: Dict[str, Any], df: pd.DataFrame) -> Path:
        """Generate detailed evaluation report"""
        report = []
        report.append("# Trained RL Models Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Philosophers: {self.n_philosophers}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        if not df.empty:
            best_eating = df.loc[df['Mean_Eating_Events'].idxmax()]
            best_fairness = df.loc[df['Mean_Fairness'].idxmax()]
            best_reward = df.loc[df['Mean_Reward'].idxmax()]
            best_efficiency = df.loc[df['Efficiency'].idxmax()]
            
            report.append(f"**Best Eating Performance:** {best_eating['Algorithm']} ({best_eating['Mean_Eating_Events']:.2f} events)")
            report.append(f"**Best Fairness:** {best_fairness['Algorithm']} ({best_fairness['Mean_Fairness']:.3f})")
            report.append(f"**Best Reward:** {best_reward['Algorithm']} ({best_reward['Mean_Reward']:.2f})")
            report.append(f"**Best Efficiency:** {best_efficiency['Algorithm']} ({best_efficiency['Efficiency']:.4f})")
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Algorithm Performance")
        report.append("")
        
        for algorithm, result in results.items():
            if 'error' in result:
                report.append(f"### {algorithm.upper()}")
                report.append(f"**Error:** {result['error']}")
                report.append("")
                continue
            
            report.append(f"### {algorithm.upper()}")
            report.append(f"**Model Path:** {result['model_path']}")
            report.append(f"**Episodes Evaluated:** {result['n_episodes']}")
            report.append(f"**Evaluation Duration:** {result['evaluation_duration']:.2f}s")
            report.append("")
            
            report.append("**Performance Metrics:**")
            report.append(f"- Eating Events: {result['mean_total_eating_events']:.2f} ± {result['std_total_eating_events']:.2f}")
            report.append(f"- Fairness Index: {result['mean_fairness']:.3f} ± {result['std_fairness']:.3f}")
            report.append(f"- Episode Reward: {result['mean_episode_reward']:.2f} ± {result['std_episode_reward']:.2f}")
            report.append(f"- Episode Length: {result['mean_episode_length']:.1f} ± {result['std_episode_length']:.1f}")
            report.append(f"- Success Rate: {result['success_rate']:.1%}")
            report.append(f"- Efficiency: {result['mean_efficiency']:.4f} ± {result['std_efficiency']:.4f}")
            report.append("")
            
            # Eating distribution
            dist = result['eating_distribution']
            report.append("**Eating Distribution per Philosopher:**")
            for i, (mean, std) in enumerate(zip(dist['mean_per_philosopher'], dist['std_per_philosopher'])):
                report.append(f"- Philosopher {i}: {mean:.2f} ± {std:.2f}")
            report.append("")
        
        # Performance Ranking
        if not df.empty:
            report.append("## Performance Rankings")
            report.append("")
            
            # Rank by different metrics
            metrics_for_ranking = ['Mean_Eating_Events', 'Mean_Fairness', 'Mean_Reward', 'Efficiency']
            
            for metric in metrics_for_ranking:
                report.append(f"### By {metric.replace('_', ' ')}")
                sorted_df = df.sort_values(metric, ascending=False)
                for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
                    report.append(f"{i}. {row['Algorithm']}: {row[metric]:.3f}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if not df.empty:
            # Find overall best performer (weighted score)
            df_normalized = df.copy()
            metrics_to_normalize = ['Mean_Eating_Events', 'Mean_Fairness', 'Efficiency']
            for metric in metrics_to_normalize:
                df_normalized[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            
            df_normalized['Overall_Score'] = (df_normalized['Mean_Eating_Events_norm'] + 
                                            df_normalized['Mean_Fairness_norm'] + 
                                            df_normalized['Efficiency_norm']) / 3
            
            best_overall = df_normalized.loc[df_normalized['Overall_Score'].idxmax()]
            
            report.append(f"**Recommended Algorithm:** {best_overall['Algorithm']}")
            report.append(f"- Provides balanced performance across eating events, fairness, and efficiency")
            report.append(f"- Overall normalized score: {best_overall['Overall_Score']:.3f}")
            report.append("")
            
            # Specific use case recommendations
            report.append("**Use Case Recommendations:**")
            
            max_eating = df.loc[df['Mean_Eating_Events'].idxmax()]
            report.append(f"- For maximum throughput: {max_eating['Algorithm']}")
            
            max_fairness = df.loc[df['Mean_Fairness'].idxmax()]
            report.append(f"- For fairest distribution: {max_fairness['Algorithm']}")
            
            max_efficiency = df.loc[df['Efficiency'].idxmax()]
            report.append(f"- For highest efficiency: {max_efficiency['Algorithm']}")
            
        report.append("")
        report.append("---")
        report.append("*Report generated by Trained RL Evaluation System*")
        
        # Save report
        report_path = self.results_dir / 'rl_evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return report_path
    
    def save_results(self, results: Dict[str, Any], df: pd.DataFrame):
        """Save all evaluation results"""
        # Save raw results
        results_to_save = {}
        for algo, result in results.items():
            # Remove detailed_results to reduce file size
            result_copy = result.copy()
            if 'detailed_results' in result_copy:
                del result_copy['detailed_results']
            results_to_save[algo] = result_copy
        
        eval_data = {
            'evaluation_results': results_to_save,
            'timestamp': time.time(),
            'n_philosophers': self.n_philosophers,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'rl_evaluation_results.json', 'w') as f:
            json.dump(eval_data, f, indent=2, default=str)
        
        # Save summary dataframe
        df.to_csv(self.results_dir / 'rl_performance_summary.csv', index=False)
        
        # Save detailed results separately
        for algo, result in results.items():
            if 'detailed_results' in result:
                detailed_path = self.results_dir / f'{algo}_detailed_results.json'
                with open(detailed_path, 'w') as f:
                    json.dump(result['detailed_results'], f, indent=2, default=str)
        
        print(f"Results saved to {self.results_dir}")
    
    def run_interactive_evaluation(self):
        """Run interactive evaluation with user choices"""
        print("RL Models Interactive Evaluation")
        print("=" * 40)
        
        models = self.find_trained_models()
        if not models:
            print("No trained models found!")
            return
        
        print(f"Available models: {list(models.keys())}")
        
        # Get user preferences
        while True:
            try:
                n_episodes = int(input(f"Number of episodes to evaluate (default 50): ") or "50")
                break
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                max_steps = int(input(f"Max steps per episode (default 1000): ") or "1000")
                break
            except ValueError:
                print("Please enter a valid number")
        
        render_demo = input("Show demo episode? (y/n): ").lower().startswith('y')
        
        # Run evaluation
        results = self.evaluate_all(n_episodes, max_steps)
        
        if not results:
            print("No results to analyze!")
            return
        
        # Show demo if requested
        if render_demo:
            print("\nRunning demo episodes...")
            for algo in list(results.keys())[:2]:  # Show max 2 demos
                if 'error' not in results[algo]:
                    print(f"\nDemo: {algo.upper()}")
                    model_path = models[algo]
                    agent, env = self.load_agent_and_env(algo, model_path)
                    self.evaluate_single_episode(agent, env, algo, max_steps=200, render=True)
        
        # Generate analysis
        df = self.create_performance_dataframe(results)
        print("\nPerformance Summary:")
        print(df.to_string(index=False))
        
        # Create visualizations
        self.plot_performance_comparison(df)
        self.plot_eating_distribution(results)
        
        # Generate and save report
        report_path = self.generate_detailed_report(results, df)
        self.save_results(results, df)
        
        print(f"\nEvaluation complete!")
        print(f"Report saved to: {report_path}")


def main():
    """Main evaluation function"""
    print("Trained RL Models Evaluation")
    print("=" * 30)
    
    evaluator = TrainedRLEvaluator()
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        evaluator.run_interactive_evaluation()
    else:
        # Standard evaluation
        results = evaluator.evaluate_all(n_episodes=50, max_steps=1000)
        
        if not results:
            print("No trained models found or evaluation failed!")
            return
        
        # Generate analysis
        df = evaluator.create_performance_dataframe(results)
        print("\nPerformance Summary:")
        print(df)
        
        # Create visualizations
        evaluator.plot_performance_comparison(df)
        evaluator.plot_eating_distribution(results)
        
        # Generate report
        report_path = evaluator.generate_detailed_report(results, df)
        evaluator.save_results(results, df)
        
        print(f"\nEvaluation complete! Report: {report_path}")


if __name__ == "__main__":
    main()