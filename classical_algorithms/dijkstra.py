"""
Dijkstra's Dining Philosophers Solution
Prevents deadlock by requiring philosophers to pick up forks in order
"""

import numpy as np
import time
from typing import Dict, Any, List


class DijkstraPhilosopher:
    """Individual philosopher using Dijkstra's algorithm"""
    
    def __init__(self, id: int, n_philosophers: int):
        self.id = id
        self.n_philosophers = n_philosophers
        self.state = "thinking"
        self.eating_count = 0
        self.left_fork = id
        self.right_fork = (id + 1) % n_philosophers
        
        # Dijkstra's ordering: lower numbered fork first
        self.first_fork = min(self.left_fork, self.right_fork)
        self.second_fork = max(self.left_fork, self.right_fork)
    
    def want_to_eat(self, forks: List[bool]) -> List[int]:
        """Return actions needed to start eating"""
        actions = []
        
        if self.state == "thinking":
            # Try to pick up first fork (lower numbered)
            if forks[self.first_fork]:
                actions.append(('pick', self.first_fork))
                self.state = "has_first"
        
        elif self.state == "has_first":
            # Try to pick up second fork
            if forks[self.second_fork]:
                actions.append(('pick', self.second_fork))
                self.state = "eating"
        
        return actions
    
    def finish_eating(self) -> List[int]:
        """Finish eating and put down forks"""
        actions = []
        if self.state == "eating":
            # Put down both forks
            actions.append(('drop', self.first_fork))
            actions.append(('drop', self.second_fork))
            self.eating_count += 1
            self.state = "thinking"
        
        return actions


class DijkstraAlgorithm:
    """Dijkstra's solution for dining philosophers"""
    
    def __init__(self, n_philosophers: int = 5):
        self.n_philosophers = n_philosophers
        self.philosophers = [DijkstraPhilosopher(i, n_philosophers) for i in range(n_philosophers)]
        self.forks = [True] * n_philosophers  # True = available
        self.step_count = 0
        self.eating_durations = [0] * n_philosophers
        self.eating_time = 3  # steps to eat
        
    def reset(self) -> Dict[str, Any]:
        """Reset to initial state"""
        self.philosophers = [DijkstraPhilosopher(i, self.n_philosophers) for i in range(self.n_philosophers)]
        self.forks = [True] * self.n_philosophers
        self.step_count = 0
        self.eating_durations = [0] * self.n_philosophers
        return self.get_state()
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the algorithm"""
        self.step_count += 1
        
        # Process eating philosophers
        for i, phil in enumerate(self.philosophers):
            if phil.state == "eating":
                self.eating_durations[i] += 1
                if self.eating_durations[i] >= self.eating_time:
                    # Finish eating
                    actions = phil.finish_eating()
                    for action_type, fork_id in actions:
                        if action_type == 'drop':
                            self.forks[fork_id] = True
                    self.eating_durations[i] = 0
        
        # Process thinking philosophers trying to eat
        for phil in self.philosophers:
            if phil.state in ["thinking", "has_first"]:
                actions = phil.want_to_eat(self.forks)
                for action_type, fork_id in actions:
                    if action_type == 'pick':
                        self.forks[fork_id] = False
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'step': self.step_count,
            'philosopher_states': [p.state for p in self.philosophers],
            'eating_counts': [p.eating_count for p in self.philosophers],
            'forks_available': self.forks.copy(),
            'currently_eating': sum(1 for p in self.philosophers if p.state == "eating"),
            'deadlocked': False  # Dijkstra prevents deadlock
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
            'algorithm': 'dijkstra',
            'final_eating_counts': eating_counts,
            'total_eating_events': sum(eating_counts),
            'eating_variance': np.var(eating_counts),
            'fairness_index': 1.0 / (1.0 + np.var(eating_counts)),
            'steps_taken': self.step_count,
            'deadlock_occurred': False,
            'states_history': states
        }