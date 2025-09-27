import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List, Any


class ContinuousDiningPhilosophersEnv(gym.Env):
    """
    Continuous Dining Philosophers Gym Environment
    
    Each philosopher has continuous actions:
    - Action[0]: Thinking intensity (0-1)
    - Action[1]: Left fork grabbing force (0-1) 
    - Action[2]: Right fork grabbing force (0-1)
    - Action[3]: Eating intensity (0-1)
    """
    
    def __init__(self, n_philosophers: int = 5, max_steps: int = 1000):
        super().__init__()
        
        self.n_philosophers = n_philosophers
        self.max_steps = max_steps
        
        # Action space: 4 continuous actions per philosopher
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_philosophers, 4),
            dtype=np.float32
        )
        
        # Observation space: [philosopher_states, fork_forces, hunger_levels, eating_progress, step_count]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_philosophers * 5 + 1,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Philosopher energy levels (0-1)
        self.energy_levels = np.ones(self.n_philosophers, dtype=np.float32)
        
        # Hunger levels (0-1, higher = more hungry)
        self.hunger_levels = np.random.uniform(0.2, 0.4, self.n_philosophers).astype(np.float32)
        
        # Fork grip strength for each philosopher (0-1)
        self.fork_grips = np.zeros((self.n_philosophers, 2), dtype=np.float32)  # [left, right]
        
        # Eating progress (0-1, 1 = finished eating)
        self.eating_progress = np.zeros(self.n_philosophers, dtype=np.float32)
        
        # Eating counts
        self.eating_counts = np.zeros(self.n_philosophers, dtype=np.int32)
        
        # Step counter
        self.step_count = 0
        
        # Fork availability (based on competing grips)
        self.fork_availability = np.ones(self.n_philosophers, dtype=np.float32)
        
        return self._get_observation()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Normalize step count
        step_progress = min(self.step_count / self.max_steps, 1.0)
        
        # Process actions
        total_reward = 0
        
        # Update fork competition
        self._update_fork_competition(actions)
        
        # Process each philosopher's actions
        for i in range(self.n_philosophers):
            reward = self._process_philosopher_actions(i, actions[i])
            total_reward += reward
        
        # Update hunger levels (increase over time)
        self.hunger_levels = np.clip(
            self.hunger_levels + np.random.uniform(0.01, 0.03, self.n_philosophers),
            0.0, 1.0
        )
        
        # Update energy based on thinking vs eating
        for i in range(self.n_philosophers):
            if self.eating_progress[i] > 0:
                # Recover energy while eating
                self.energy_levels[i] = min(1.0, self.energy_levels[i] + 0.05)
                # Reduce hunger while eating
                self.hunger_levels[i] = max(0.0, self.hunger_levels[i] - 0.1)
            else:
                # Lose energy while thinking/waiting
                self.energy_levels[i] = max(0.0, self.energy_levels[i] - 0.01)
        
        # Add fairness reward/penalty
        if np.sum(self.eating_counts) > 0:
            eating_variance = np.var(self.eating_counts)
            fairness_penalty = -eating_variance * 0.1
            total_reward += fairness_penalty
        
        # Episode termination
        done = (self.step_count >= self.max_steps or 
                np.mean(self.energy_levels) < 0.1)  # philosophers too exhausted
        
        info = {
            'eating_counts': self.eating_counts.copy(),
            'hunger_levels': self.hunger_levels.copy(),
            'energy_levels': self.energy_levels.copy(),
            'fork_grips': self.fork_grips.copy(),
            'eating_progress': self.eating_progress.copy(),
            'step_progress': step_progress
        }
        
        return self._get_observation(), total_reward, done, info
    
    def _update_fork_competition(self, actions: np.ndarray):
        """Update fork availability based on competing grips"""
        # Calculate grip strengths for each fork
        fork_competitors = np.zeros((self.n_philosophers, 2))  # [left_grip, right_grip] for each fork
        
        for i in range(self.n_philosophers):
            left_fork_idx = i
            right_fork_idx = (i + 1) % self.n_philosophers
            
            # Philosopher i competes for left fork (fork i) and right fork (fork i+1)
            left_grip = actions[i][1] * self.energy_levels[i]  # left fork action * energy
            right_grip = actions[i][2] * self.energy_levels[i]  # right fork action * energy
            
            # Update grip strengths
            self.fork_grips[i][0] = left_grip
            self.fork_grips[i][1] = right_grip
        
        # Determine fork winners based on grip competition
        for fork_id in range(self.n_philosophers):
            # Two philosophers compete for each fork
            left_philosopher = (fork_id - 1) % self.n_philosophers
            right_philosopher = fork_id
            
            left_grip = self.fork_grips[left_philosopher][1]  # right grip of left philosopher
            right_grip = self.fork_grips[right_philosopher][0]  # left grip of right philosopher
            
            # Fork goes to stronger grip, or shared if similar
            total_grip = left_grip + right_grip
            if total_grip > 0:
                self.fork_availability[fork_id] = max(left_grip, right_grip) / total_grip
            else:
                self.fork_availability[fork_id] = 1.0
    
    def _process_philosopher_actions(self, philosopher_id: int, action: np.ndarray) -> float:
        """Process actions for a specific philosopher"""
        reward = 0
        
        thinking_intensity = action[0]
        left_grab = action[1]
        right_grab = action[2] 
        eating_intensity = action[3]
        
        left_fork = philosopher_id
        right_fork = (philosopher_id + 1) % self.n_philosophers
        
        # Check if philosopher has both forks
        left_fork_strength = self.fork_grips[philosopher_id][0]
        right_fork_strength = self.fork_grips[philosopher_id][1]
        
        has_left_fork = (left_fork_strength > 0.5 and 
                        self.fork_availability[left_fork] > 0.7)
        has_right_fork = (right_fork_strength > 0.5 and 
                         self.fork_availability[right_fork] > 0.7)
        
        has_both_forks = has_left_fork and has_right_fork
        
        # Thinking reward
        if thinking_intensity > 0.5:
            reward += thinking_intensity * 0.1 * self.energy_levels[philosopher_id]
        
        # Fork grabbing penalties (energy cost)
        fork_grab_cost = (left_grab + right_grab) * 0.05
        reward -= fork_grab_cost
        
        # Eating logic
        if has_both_forks and eating_intensity > 0.5:
            # Can eat
            eating_progress_delta = eating_intensity * 0.2 * self.energy_levels[philosopher_id]
            self.eating_progress[philosopher_id] += eating_progress_delta
            
            # Reward for eating
            eating_reward = eating_intensity * self.hunger_levels[philosopher_id] * 3.0
            reward += eating_reward
            
            # Check if finished eating
            if self.eating_progress[philosopher_id] >= 1.0:
                self.eating_counts[philosopher_id] += 1
                self.eating_progress[philosopher_id] = 0.0
                self.hunger_levels[philosopher_id] = max(0.0, self.hunger_levels[philosopher_id] - 0.5)
                reward += 5.0  # completion bonus
        
        elif eating_intensity > 0.5:
            # Trying to eat without both forks
            reward -= 2.0
        
        # Hunger penalty
        hunger_penalty = self.hunger_levels[philosopher_id] * 0.5
        reward -= hunger_penalty
        
        # Energy depletion penalty
        if self.energy_levels[philosopher_id] < 0.3:
            reward -= 1.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Flatten fork grips
        fork_grips_flat = self.fork_grips.flatten()
        
        obs = np.concatenate([
            self.energy_levels,
            self.hunger_levels,
            fork_grips_flat,
            self.eating_progress,
            [min(self.step_count / self.max_steps, 1.0)]
        ])
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render the current state"""
        if mode == 'human':
            print(f"\nStep {self.step_count}")
            
            print("Philosophers:")
            for i in range(self.n_philosophers):
                energy = self.energy_levels[i]
                hunger = self.hunger_levels[i]
                eating = self.eating_progress[i]
                left_grip = self.fork_grips[i][0]
                right_grip = self.fork_grips[i][1]
                
                status = "EATING" if eating > 0 else "THINKING"
                print(f"  P{i}: {status} | Energy:{energy:.2f} Hunger:{hunger:.2f} "
                      f"L-grip:{left_grip:.2f} R-grip:{right_grip:.2f} Eat-prog:{eating:.2f}")
            
            print("\nFork availability:")
            for i, avail in enumerate(self.fork_availability):
                print(f"  F{i}: {avail:.2f}")
            
            print(f"Eating counts: {self.eating_counts}")
            print("-" * 70)