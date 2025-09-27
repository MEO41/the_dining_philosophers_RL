"""
Waiter Solution for Dining Philosophers
A central waiter controls fork allocation to prevent deadlock
"""

import numpy as np
import random
from typing import Dict, Any, List, Optional


class WaiterPhilosopher:
    """Philosopher in waiter solution"""
    
    def __init__(self, id: int):
        self.id = id
        self.state = "thinking"
        self.eating_count = 0
        self.hunger_level = random.uniform(0.3, 0.7)
        self.thinking_time = 0
        self.eating_time = 0
    
    def update_hunger(self):
        """Update hunger level"""
        if self.state == "thinking":
            self.thinking_time += 1
            self.hunger_level = min(1.0, self.hunger_level + 0.05)
        elif self.state == "eating":
            self.eating_time += 1
            self.hunger_level = max(0.0, self.hunger_level - 0.1)


class Waiter:
    """Central waiter controlling fork allocation"""
    
    def __init__(self, n_philosophers: int):
        self.n_philosophers = n_philosophers
        self.forks = [True] * n_philosophers  # True = available
        self.fork_requests = []  # Queue of (philosopher_id, priority)
        self.eating_permissions = set()  # Philosophers allowed to eat
        self.max_concurrent_eaters = max(1, (n_philosophers - 1) // 2)
    
    def request_forks(self, philosopher_id: int, hunger_level: float):
        """Philosopher requests permission to eat"""
        if philosopher_id not in [req[0] for req in self.fork_requests]:
            priority = hunger_level  # Higher hunger = higher priority
            self.fork_requests.append((philosopher_id, priority))
    
    def can_eat_safely(self, philosopher_id: int) -> bool:
        """Check if philosopher can eat without causing deadlock"""
        left_fork = philosopher_id
        right_fork = (philosopher_id + 1) % self.n_philosophers
        
        # Check if both forks are available
        if not (self.forks[left_fork] and self.forks[right_fork]):
            return False
        
        # Check if granting permission would exceed max concurrent eaters
        if len(self.eating_permissions) >= self.max_concurrent_eaters:
            return False
        
        # Check if neighbors are eating (additional safety)
        left_neighbor = (philosopher_id - 1) % self.n_philosophers
        right_neighbor = (philosopher_id + 1) % self.n_philosophers
        
        if left_neighbor in self.eating_permissions or right_neighbor in self.eating_permissions:
            return False
        
        return True
    
    def allocate_forks(self):
        """Process fork requests and allocate based on priority"""
        # Sort requests by priority (hunger level)
        self.fork_requests.sort(key=lambda x: x[1], reverse=True)
        
        granted = []
        for philosopher_id, priority in self.fork_requests:
            if self.can_eat_safely(philosopher_id):
                # Grant permission
                left_fork = philosopher_id
                right_fork = (philosopher_id + 1) % self.n_philosophers
                
                self.forks[left_fork] = False
                self.forks[right_fork] = False
                self.eating_permissions.add(philosopher_id)
                granted.append(philosopher_id)
        
        # Remove granted requests
        self.fork_requests = [(pid, pri) for pid, pri in self.fork_requests if pid not in granted]
    
    def release_forks(self, philosopher_id: int):
        """Release forks when philosopher finishes eating"""
        if philosopher_id in self.eating_permissions:
            left_fork = philosopher_id
            right_fork = (philosopher_id + 1) % self.n_philosophers
            
            self.forks[left_fork] = True
            self.forks[right_fork] = True
            self.eating_permissions.remove(philosopher_id)


class WaiterAlgorithm:
    """Waiter solution for dining philosophers"""
    
    def __init__(self, n_philosophers: int = 5):
        self.n_philosophers = n_philosophers
        self.philosophers = [WaiterPhilosopher(i) for i in range(n_philosophers)]
        self.waiter = Waiter(n_philosophers)
        self.step_count = 0
        self.eating_durations = [0] * n_philosophers
        self.eating_time = 3
    
    def reset(self) -> Dict[str, Any]:
        """Reset to initial state"""
        self.philosophers = [WaiterPhilosopher(i) for i in range(self.n_philosophers)]
        self.waiter = Waiter(self.n_philosophers)
        self.step_count = 0
        self.eating_durations = [0] * self.n_philosophers
        return self.get_state()
    
    def step(self) -> Dict[str, Any]:
        """Execute one step"""
        self.step_count += 1
        
        # Update philosopher states
        for i, phil in enumerate(self.philosophers):
            phil.update_hunger()
            
            if phil.state == "eating":
                self.eating_durations[i] += 1
                if self.eating_durations[i] >= self.eating_time:
                    # Finish eating
                    phil.state = "thinking"
                    phil.eating_count += 1
                    phil.eating_time = 0
                    self.eating_durations[i] = 0
                    self.waiter.release_forks(i)
            
            elif phil.state == "thinking":
                # Request to eat based on hunger
                if phil.hunger_level > 0.6:  # Hungry threshold
                    self.waiter.request_forks(i, phil.hunger_level)
        
        # Waiter processes requests
        self.waiter.allocate_forks()
        
        # Update states based on permissions
        for i, phil in enumerate(self.philosophers):
            if i in self.waiter.eating_permissions and phil.state == "thinking":
                phil.state = "eating"
                phil.thinking_time = 0
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'step': self.step_count,
            'philosopher_states': [p.state for p in self.philosophers],
            'eating_counts': [p.eating_count for p in self.philosophers],
            'hunger_levels': [p.hunger_level for p in self.philosophers],
            'forks_available': self.waiter.forks.copy(),
            'currently_eating': len(self.waiter.eating_permissions),
            'pending_requests': len(self.waiter.fork_requests),
            'deadlocked': False  # Waiter prevents deadlock
        }
    
    def run_simulation(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run complete simulation"""
        self.reset()
        states = []
        
        for _ in range(max_steps):
            state = self.step()
            states.append(state)
            
            # Stop if all philosophers have eaten multiple times
            if all(p.eating_count >= 3 for p in self.philosophers):
                break
        
        final_state = self.get_state()
        eating_counts = final_state['eating_counts']
        
        return {
            'algorithm': 'waiter',
            'final_eating_counts': eating_counts,
            'total_eating_events': sum(eating_counts),
            'eating_variance': np.var(eating_counts),
            'fairness_index': 1.0 / (1.0 + np.var(eating_counts)),
            'steps_taken': self.step_count,
            'deadlock_occurred': False,
            'average_hunger': np.mean([p.hunger_level for p in self.philosophers]),
            'states_history': states
        }