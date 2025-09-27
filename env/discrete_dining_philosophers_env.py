import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List, Any


class DiscreteDiningPhilosophersEnv(gym.Env):
    """
    Discrete Dining Philosophers Gym Environment
    
    Each philosopher can take discrete actions:
    0: Think (do nothing)
    1: Try to pick up left fork
    2: Try to pick up right fork  
    3: Eat (if has both forks)
    4: Put down left fork
    5: Put down right fork
    """
    
    def __init__(self, n_philosophers: int = 5, max_steps: int = 1000):
        super().__init__()
        
        self.n_philosophers = n_philosophers
        self.max_steps = max_steps
        
        # Action space: each philosopher chooses from 6 actions
        self.action_space = spaces.MultiDiscrete([6] * n_philosophers)
        
        # Observation space: [philosopher_states, fork_states, eating_counts, step_count]
        # Philosopher states: 0=thinking, 1=has_left_fork, 2=has_right_fork, 3=has_both_forks, 4=eating
        # Fork states: 0=available, philosopher_id+1=taken_by_philosopher
        self.observation_space = spaces.Box(
            low=0,
            high=max(n_philosophers + 1, max_steps),
            shape=(n_philosophers * 2 + n_philosophers + 1,),
            dtype=np.int32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Philosopher states: 0=thinking
        self.philosopher_states = np.zeros(self.n_philosophers, dtype=np.int32)
        
        # Fork ownership: 0=available, philosopher_id+1=taken
        self.fork_owners = np.zeros(self.n_philosophers, dtype=np.int32)
        
        # Track eating counts for each philosopher
        self.eating_counts = np.zeros(self.n_philosophers, dtype=np.int32)
        
        # Track eating duration for each philosopher
        self.eating_duration = np.zeros(self.n_philosophers, dtype=np.int32)
        
        # Step counter
        self.step_count = 0
        
        # Deadlock detection
        self.deadlock_steps = 0
        
        return self._get_observation()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Process actions for each philosopher
        rewards = np.zeros(self.n_philosophers)
        
        for i, action in enumerate(actions):
            reward = self._execute_action(i, action)
            rewards[i] = reward
        
        # Update eating durations
        for i in range(self.n_philosophers):
            if self.philosopher_states[i] == 4:  # eating
                self.eating_duration[i] += 1
                if self.eating_duration[i] >= 3:  # finish eating after 3 steps
                    self._finish_eating(i)
        
        # Calculate total reward
        total_reward = np.sum(rewards)
        
        # Add fairness bonus/penalty
        eating_variance = np.var(self.eating_counts) if np.sum(self.eating_counts) > 0 else 0
        fairness_penalty = -eating_variance * 0.1
        total_reward += fairness_penalty
        
        # Check for deadlock
        if self._is_deadlocked():
            self.deadlock_steps += 1
            total_reward -= 10  # Heavy penalty for deadlock
        else:
            self.deadlock_steps = 0
        
        # Episode termination conditions
        done = (self.step_count >= self.max_steps or 
                self.deadlock_steps > 10)
        
        info = {
            'eating_counts': self.eating_counts.copy(),
            'deadlock_steps': self.deadlock_steps,
            'philosopher_states': self.philosopher_states.copy(),
            'fork_owners': self.fork_owners.copy()
        }
        
        return self._get_observation(), total_reward, done, info
    
    def _execute_action(self, philosopher_id: int, action: int) -> float:
        """Execute action for a specific philosopher"""
        reward = 0
        left_fork = philosopher_id
        right_fork = (philosopher_id + 1) % self.n_philosophers
        
        if action == 0:  # Think
            if self.philosopher_states[philosopher_id] != 4:  # not eating
                self.philosopher_states[philosopher_id] = 0
                reward = 0.1  # small reward for thinking
                
        elif action == 1:  # Try to pick up left fork
            if (self.philosopher_states[philosopher_id] == 0 and 
                self.fork_owners[left_fork] == 0):
                self.fork_owners[left_fork] = philosopher_id + 1
                self.philosopher_states[philosopher_id] = 1
                reward = 1
            else:
                reward = -0.5  # penalty for invalid action
                
        elif action == 2:  # Try to pick up right fork
            if (self.philosopher_states[philosopher_id] in [0, 1] and 
                self.fork_owners[right_fork] == 0):
                self.fork_owners[right_fork] = philosopher_id + 1
                if self.philosopher_states[philosopher_id] == 1:
                    self.philosopher_states[philosopher_id] = 3  # has both forks
                else:
                    self.philosopher_states[philosopher_id] = 2  # has right fork only
                reward = 1
            else:
                reward = -0.5
                
        elif action == 3:  # Eat
            if self.philosopher_states[philosopher_id] == 3:  # has both forks
                self.philosopher_states[philosopher_id] = 4  # eating
                self.eating_duration[philosopher_id] = 0
                reward = 5  # high reward for eating
            else:
                reward = -1  # penalty for trying to eat without both forks
                
        elif action == 4:  # Put down left fork
            if self.fork_owners[left_fork] == philosopher_id + 1:
                self.fork_owners[left_fork] = 0
                if self.philosopher_states[philosopher_id] == 3:
                    self.philosopher_states[philosopher_id] = 2  # now has only right
                elif self.philosopher_states[philosopher_id] == 1:
                    self.philosopher_states[philosopher_id] = 0  # thinking
                reward = 0.5
            else:
                reward = -0.5
                
        elif action == 5:  # Put down right fork
            if self.fork_owners[right_fork] == philosopher_id + 1:
                self.fork_owners[right_fork] = 0
                if self.philosopher_states[philosopher_id] == 3:
                    self.philosopher_states[philosopher_id] = 1  # now has only left
                elif self.philosopher_states[philosopher_id] == 2:
                    self.philosopher_states[philosopher_id] = 0  # thinking
                reward = 0.5
            else:
                reward = -0.5
        
        return reward
    
    def _finish_eating(self, philosopher_id: int):
        """Handle philosopher finishing eating"""
        self.eating_counts[philosopher_id] += 1
        self.eating_duration[philosopher_id] = 0
        
        # Put down both forks
        left_fork = philosopher_id
        right_fork = (philosopher_id + 1) % self.n_philosophers
        
        self.fork_owners[left_fork] = 0
        self.fork_owners[right_fork] = 0
        self.philosopher_states[philosopher_id] = 0  # back to thinking
    
    def _is_deadlocked(self) -> bool:
        """Check if system is in deadlock"""
        # Simple deadlock detection: all philosophers holding exactly one fork
        holding_one_fork = sum(1 for state in self.philosopher_states if state in [1, 2])
        return holding_one_fork == self.n_philosophers
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = np.concatenate([
            self.philosopher_states,
            self.fork_owners,
            self.eating_counts,
            [self.step_count]
        ])
        return obs.astype(np.int32)
    
    def render(self, mode='human'):
        """Render the current state"""
        if mode == 'human':
            print(f"\nStep {self.step_count}")
            print("Philosophers:", end=" ")
            state_names = {0: "T", 1: "L", 2: "R", 3: "B", 4: "E"}
            for i, state in enumerate(self.philosopher_states):
                print(f"P{i}:{state_names[state]}", end=" ")
            print()
            
            print("Forks:", end=" ")
            for i, owner in enumerate(self.fork_owners):
                if owner == 0:
                    print(f"F{i}:free", end=" ")
                else:
                    print(f"F{i}:P{owner-1}", end=" ")
            print()
            
            print("Eating counts:", self.eating_counts)
            print("-" * 50)