#!/usr/bin/env python3
"""
Test script for dining philosophers environments
Usage: python test/test_env.py <env_type>
Where env_type is either 'discrete' or 'continuous'
"""

import sys
import os
import numpy as np

# Add the env directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env'))

from discrete_dining_philosophers_env import DiscreteDiningPhilosophersEnv
from continuous_dining_philosophers_env import ContinuousDiningPhilosophersEnv


def test_discrete_env():
    """Test the discrete dining philosophers environment"""
    print("=== Testing Discrete Dining Philosophers Environment ===\n")
    
    env = DiscreteDiningPhilosophersEnv(n_philosophers=5, max_steps=100)
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test random actions
    print("\nTesting random actions for 20 steps...")
    total_reward = 0
    
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"Action: {action}")
            print(f"Reward: {reward:.2f}")
            print(f"Done: {done}")
            print(f"Eating counts: {info['eating_counts']}")
            env.render()
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    # Test specific action sequences
    print("\n=== Testing Specific Action Sequences ===")
    env.reset()
    
    # Test philosopher 0 trying to eat
    actions = np.zeros(5, dtype=int)
    
    print("\nPhilosopher 0 picks up left fork:")
    actions[0] = 1  # pick up left fork
    obs, reward, done, info = env.step(actions)
    print(f"Reward: {reward:.2f}, Philosopher 0 state: {info['philosopher_states'][0]}")
    env.render()
    
    print("\nPhilosopher 0 picks up right fork:")
    actions[0] = 2  # pick up right fork
    obs, reward, done, info = env.step(actions)
    print(f"Reward: {reward:.2f}, Philosopher 0 state: {info['philosopher_states'][0]}")
    env.render()
    
    print("\nPhilosopher 0 tries to eat:")
    actions[0] = 3  # eat
    obs, reward, done, info = env.step(actions)
    print(f"Reward: {reward:.2f}, Philosopher 0 state: {info['philosopher_states'][0]}")
    env.render()
    
    # Let philosopher eat for a few steps
    for _ in range(3):
        actions[0] = 0  # think (wait)
        obs, reward, done, info = env.step(actions)
        env.render()
        print(f"Eating counts after step: {info['eating_counts']}")
    
    print("\n=== Testing Deadlock Scenario ===")
    env.reset()
    
    # All philosophers try to pick up left fork simultaneously
    actions = np.ones(5, dtype=int)  # all pick up left fork
    
    for step in range(5):
        obs, reward, done, info = env.step(actions)
        print(f"Step {step}: Deadlock steps: {info['deadlock_steps']}, Reward: {reward:.2f}")
        env.render()
        
        if info['deadlock_steps'] > 0:
            print("Deadlock detected!")
            break


def test_continuous_env():
    """Test the continuous dining philosophers environment"""
    print("=== Testing Continuous Dining Philosophers Environment ===\n")
    
    env = ContinuousDiningPhilosophersEnv(n_philosophers=5, max_steps=100)
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test random actions
    print("\nTesting random actions for 20 steps...")
    total_reward = 0
    
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"Reward: {reward:.2f}")
            print(f"Done: {done}")
            print(f"Eating counts: {info['eating_counts']}")
            print(f"Average hunger: {np.mean(info['hunger_levels']):.2f}")
            print(f"Average energy: {np.mean(info['energy_levels']):.2f}")
            env.render()
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    # Test specific action sequences
    print("\n=== Testing Coordinated Actions ===")
    env.reset()
    
    # Test philosopher 0 trying to eat with high intensity
    for step in range(10):
        actions = np.zeros((5, 4))
        
        # Philosopher 0: high grab forces and eating intensity
        actions[0] = [0.1, 0.9, 0.9, 0.8]  # [thinking, left_grab, right_grab, eating]
        
        # Other philosophers: mostly thinking
        for i in range(1, 5):
            actions[i] = [0.8, 0.1, 0.1, 0.0]
        
        obs, reward, done, info = env.step(actions)
        
        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"Reward: {reward:.2f}")
            print(f"P0 eating progress: {info['eating_progress'][0]:.2f}")
            print(f"P0 hunger: {info['hunger_levels'][0]:.2f}")
            env.render()
        
        if info['eating_counts'][0] > 0:
            print(f"Philosopher 0 completed eating at step {step}!")
            break
    
    print("\n=== Testing Competition Scenario ===")
    env.reset()
    
    # All philosophers compete for forks
    for step in range(15):
        actions = np.random.uniform(0.3, 0.8, (5, 4))  # moderate to high actions
        
        obs, reward, done, info = env.step(actions)
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"Reward: {reward:.2f}")
            print(f"Total eating events: {np.sum(info['eating_counts'])}")
            print(f"Eating variance: {np.var(info['eating_counts']):.2f}")
            env.render()


def run_benchmark(env_type: str, n_episodes: int = 5):
    """Run benchmark tests"""
    print(f"\n=== Benchmark: {env_type.upper()} Environment ===")
    
    if env_type == 'discrete':
        env = DiscreteDiningPhilosophersEnv(n_philosophers=5, max_steps=200)
    else:
        env = ContinuousDiningPhilosophersEnv(n_philosophers=5, max_steps=200)
    
    episode_rewards = []
    episode_eating_counts = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            if env_type == 'discrete':
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_eating_counts.append(info['eating_counts'])
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={step_count}, "
              f"Total eating events={np.sum(info['eating_counts'])}")
    
    print(f"\nBenchmark Results:")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average eating events per episode: {np.mean([np.sum(counts) for counts in episode_eating_counts]):.2f}")
    
    # Calculate fairness metrics
    all_eating_counts = np.array(episode_eating_counts)
    eating_variances = [np.var(counts) for counts in episode_eating_counts]
    print(f"Average eating variance (fairness): {np.mean(eating_variances):.2f}")


def main():
    """Main test function"""
    if len(sys.argv) != 2:
        print("Usage: python test_env.py <env_type>")
        print("env_type should be 'discrete' or 'continuous'")
        sys.exit(1)
    
    env_type = sys.argv[1].lower()
    
    if env_type == 'discrete':
        test_discrete_env()
    elif env_type == 'continuous':
        test_continuous_env()
    else:
        print(f"Unknown environment type: {env_type}")
        print("Use 'discrete' or 'continuous'")
        sys.exit(1)
    
    # Run benchmark
    run_benchmark(env_type)
    
    print(f"\n=== {env_type.upper()} Environment Test Complete ===")


if __name__ == "__main__":
    main()