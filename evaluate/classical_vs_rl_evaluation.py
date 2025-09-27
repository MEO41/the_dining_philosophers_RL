#!/usr/bin/env python3
"""
Classical vs RL Algorithms Evaluation for Dining Philosophers
Compares performance of classical algorithms against trained RL agents
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import json
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rl_models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'classical_algorithms'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

# Classical algorithms
from dijkstra import DijkstraAlgorithm
from waiter import WaiterAlgorithm
from chandy_mistra import ChandyMistraAlgorithm

# RL environments (for evaluation)
from discrete_dining_philosophers_env import DiscreteDiningPhilosophersEnv
from continuous_dining_philosophers_env import ContinuousDiningPhilosophersEnv


class ClassicalEvaluator:
    """Evaluator for classical algorithms"""
    
    def __init__(self, n_philosophers: int = 5):
        self.n_philosophers = n_philosophers
        self.algorithms = {
            'dijkstra': DijkstraAlgorithm(n_philosophers),
            'waiter': WaiterAlgorithm(n_philosophers),
            'chandy-mistra': ChandyMistraAlgorithm(n_philosophers)
        }
    
    def evaluate_algorithm(self, algorithm_name: str, n_runs: int = 10, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate a classical algorithm"""
        algorithm = self.algorithms[algorithm_name]
        results = []
        
        for run in range(n_runs):
            result = algorithm.run_simulation(max_steps)
            results.append(result)
        
        # Aggregate results
        eating_counts = [r['final_eating_counts'] for r in results]
        total_events = [r['total_eating_events'] for r in results]
        fairness_scores = [r['fairness_index'] for r in results]
        steps_taken = [r['steps_taken'] for r in results]
        deadlocks = [r['deadlock_occurred'] for r in results]
        
        return {
            'algorithm': algorithm_name,
            'n_runs': n_runs,
            'mean_eating_events': np.mean(total_events),
            'std_eating_events': np.std(total_events),
            'mean_fairness': np.mean(fairness_scores),
            'std_fairness': np.std(fairness_scores),
            'mean_steps': np.mean(steps_taken),
            'std_steps': np.std(steps_taken),
            'deadlock_rate': np.mean(deadlocks),
            'eating_counts_per_run': eating_counts,
            'detailed_results': results
        }
    
    def evaluate_all(self, n_runs: int = 10, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate all classical algorithms"""
        results = {}
        
        print("Evaluating Classical Algorithms...")
        for name in self.algorithms.keys():
            print(f"  Running {name}...")
            start_time = time.time()
            results[name] = self.evaluate_algorithm(name, n_runs, max_steps)
            duration = time.time() - start_time
            print(f"    Completed in {duration:.2f}s")
        
        return results


class RLEvaluator:
    """Evaluator for trained RL agents"""
    
    def __init__(self, n_philosophers: int = 5):
        self.n_philosophers = n_philosophers
        self.models_dir = Path("models")
    
    def load_trained_agent(self, algorithm: str) -> Any:
        """Load trained RL agent"""
        # Find the best model for the algorithm
        algo_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and algorithm in d.name]
        
        if not algo_dirs:
            raise FileNotFoundError(f"No trained {algorithm} model found")
        
        # Use most recent directory
        latest_dir = max(algo_dirs, key=lambda x: x.stat().st_mtime)
        best_model_path = latest_dir / f"{algorithm}_model_best.pth"
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_model_path}")
        
        # Load agent based on algorithm
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
        
        agent.load(str(best_model_path))
        return agent, env
    
    def evaluate_rl_agent(self, algorithm: str, n_runs: int = 10, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate trained RL agent"""
        try:
            agent, env = self.load_trained_agent(algorithm)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            return {'algorithm': algorithm, 'error': str(e)}
        
        results = []
        
        for run in range(n_runs):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
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
                
                if done:
                    break
            
            eating_counts = info['eating_counts']
            total_events = np.sum(eating_counts)
            fairness = 1.0 / (1.0 + np.var(eating_counts))
            
            results.append({
                'eating_counts': eating_counts,
                'total_events': total_events,
                'fairness': fairness,
                'steps': steps,
                'reward': episode_reward
            })
        
        # Aggregate results
        eating_counts = [r['eating_counts'] for r in results]
        total_events = [r['total_events'] for r in results]
        fairness_scores = [r['fairness'] for r in results]
        steps_taken = [r['steps'] for r in results]
        rewards = [r['reward'] for r in results]
        
        return {
            'algorithm': algorithm,
            'n_runs': n_runs,
            'mean_eating_events': np.mean(total_events),
            'std_eating_events': np.std(total_events),
            'mean_fairness': np.mean(fairness_scores),
            'std_fairness': np.std(fairness_scores),
            'mean_steps': np.mean(steps_taken),
            'std_steps': np.std(steps_taken),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'eating_counts_per_run': eating_counts,
            'detailed_results': results
        }
    
    def evaluate_all_rl(self, algorithms: List[str] = None, n_runs: int = 10, max_steps: int = 1000) -> Dict[str, Any]:
        """Evaluate all available trained RL agents"""
        if algorithms is None:
            algorithms = ['ppo', 'a2c', 'sac', 'dqn']
        
        results = {}
        
        print("Evaluating RL Algorithms...")
        for algorithm in algorithms:
            print(f"  Running {algorithm}...")
            start_time = time.time()
            results[algorithm] = self.evaluate_rl_agent(algorithm, n_runs, max_steps)
            duration = time.time() - start_time
            print(f"    Completed in {duration:.2f}s")
        
        return results


class ComparisonAnalyzer:
    """Analyzer for comparing classical and RL results"""
    
    def __init__(self):
        self.results_dir = Path("evaluate/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_comparison_dataframe(self, classical_results: Dict, rl_results: Dict) -> pd.DataFrame:
        """Create comprehensive comparison dataframe"""
        data = []
        
        # Add classical results
        for algo, result in classical_results.items():
            if 'error' not in result:
                data.append({
                    'Algorithm': algo,
                    'Type': 'Classical',
                    'Mean_Eating_Events': result['mean_eating_events'],
                    'Std_Eating_Events': result['std_eating_events'],
                    'Mean_Fairness': result['mean_fairness'],
                    'Std_Fairness': result['std_fairness'],
                    'Mean_Steps': result['mean_steps'],
                    'Std_Steps': result['std_steps'],
                    'Deadlock_Rate': result.get('deadlock_rate', 0),
                    'Mean_Reward': np.nan  # Classical algorithms don't have rewards
                })
        
        # Add RL results
        for algo, result in rl_results.items():
            if 'error' not in result:
                data.append({
                    'Algorithm': algo,
                    'Type': 'RL',
                    'Mean_Eating_Events': result['mean_eating_events'],
                    'Std_Eating_Events': result['std_eating_events'],
                    'Mean_Fairness': result['mean_fairness'],
                    'Std_Fairness': result['std_fairness'],
                    'Mean_Steps': result['mean_steps'],
                    'Std_Steps': result['std_steps'],
                    'Deadlock_Rate': 0,  # RL environments handle deadlock differently
                    'Mean_Reward': result.get('mean_reward', np.nan)
                })
        
        return pd.DataFrame(data)
    
    def plot_comparison(self, df: pd.DataFrame):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = [
            ('Mean_Eating_Events', 'Total Eating Events'),
            ('Mean_Fairness', 'Fairness Index'),
            ('Mean_Steps', 'Steps to Completion'),
            ('Std_Eating_Events', 'Eating Events Std'),
            ('Std_Fairness', 'Fairness Std'),
            ('Deadlock_Rate', 'Deadlock Rate')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            # Create grouped bar plot
            classical_data = df[df['Type'] == 'Classical']
            rl_data = df[df['Type'] == 'RL']
            
            x_pos = np.arange(len(classical_data) + len(rl_data))
            
            # Plot classical algorithms
            ax.bar(x_pos[:len(classical_data)], classical_data[metric], 
                  alpha=0.7, label='Classical', color='skyblue')
            
            # Plot RL algorithms
            ax.bar(x_pos[len(classical_data):], rl_data[metric], 
                  alpha=0.7, label='RL', color='lightcoral')
            
            ax.set_title(title)
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(metric.replace('_', ' '))
            
            # Set x-axis labels
            all_labels = list(classical_data['Algorithm']) + list(rl_data['Algorithm'])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_labels, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'classical_vs_rl_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, classical_results: Dict, rl_results: Dict, df: pd.DataFrame):
        """Generate comprehensive comparison report"""
        report = []
        report.append("# Classical vs RL Algorithms Comparison Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append("")
        
        # Best performers
        best_eating = df.loc[df['Mean_Eating_Events'].idxmax()]
        best_fairness = df.loc[df['Mean_Fairness'].idxmax()]
        fastest = df.loc[df['Mean_Steps'].idxmin()]
        
        report.append(f"**Best Eating Performance:** {best_eating['Algorithm']} ({best_eating['Type']}) - {best_eating['Mean_Eating_Events']:.2f} events")
        report.append(f"**Best Fairness:** {best_fairness['Algorithm']} ({best_fairness['Type']}) - {best_fairness['Mean_Fairness']:.3f}")
        report.append(f"**Fastest Completion:** {fastest['Algorithm']} ({fastest['Type']}) - {fastest['Mean_Steps']:.0f} steps")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        # Classical algorithms
        report.append("### Classical Algorithms")
        for algo, result in classical_results.items():
            if 'error' not in result:
                report.append(f"**{algo.title()}:**")
                report.append(f"- Eating Events: {result['mean_eating_events']:.2f} ± {result['std_eating_events']:.2f}")
                report.append(f"- Fairness: {result['mean_fairness']:.3f} ± {result['std_fairness']:.3f}")
                report.append(f"- Steps: {result['mean_steps']:.0f} ± {result['std_steps']:.0f}")
                report.append(f"- Deadlock Rate: {result.get('deadlock_rate', 0):.1%}")
                report.append("")
        
        # RL algorithms
        report.append("### RL Algorithms")
        for algo, result in rl_results.items():
            if 'error' not in result:
                report.append(f"**{algo.upper()}:**")
                report.append(f"- Eating Events: {result['mean_eating_events']:.2f} ± {result['std_eating_events']:.2f}")
                report.append(f"- Fairness: {result['mean_fairness']:.3f} ± {result['std_fairness']:.3f}")
                report.append(f"- Steps: {result['mean_steps']:.0f} ± {result['std_steps']:.0f}")
                if 'mean_reward' in result:
                    report.append(f"- Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
                report.append("")
        
        # Save report
        report_path = self.results_dir / 'comparison_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return report_path
    
    def save_results(self, classical_results: Dict, rl_results: Dict, df: pd.DataFrame):
        """Save all results to files"""
        # Save raw results
        all_results = {
            'classical': classical_results,
            'rl': rl_results,
            'timestamp': time.time()
        }
        
        with open(self.results_dir / 'comparison_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save dataframe
        df.to_csv(self.results_dir / 'comparison_summary.csv', index=False)
        
        print(f"Results saved to {self.results_dir}")


def main():
    """Main evaluation function"""
    print("Classical vs RL Algorithms Evaluation")
    print("=" * 50)
    
    n_philosophers = 5
    n_runs = 20
    max_steps = 1000
    
    # Evaluate classical algorithms
    classical_evaluator = ClassicalEvaluator(n_philosophers)
    classical_results = classical_evaluator.evaluate_all(n_runs, max_steps)
    
    # Evaluate RL algorithms
    rl_evaluator = RLEvaluator(n_philosophers)
    rl_results = rl_evaluator.evaluate_all_rl(n_runs=n_runs, max_steps=max_steps)
    
    # Analyze and compare
    analyzer = ComparisonAnalyzer()
    df = analyzer.create_comparison_dataframe(classical_results, rl_results)
    
    print("\nComparison Summary:")
    print(df)
    
    # Generate visualizations and report
    analyzer.plot_comparison(df)
    report_path = analyzer.generate_report(classical_results, rl_results, df)
    analyzer.save_results(classical_results, rl_results, df)
    
    print(f"\nEvaluation complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()