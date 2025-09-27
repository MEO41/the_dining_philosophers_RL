"""
Chandy-Mistra Drinking Philosophers Solution
Uses a token-based system with philosopher priorities
"""

import numpy as np
import random
from typing import Dict, Any, List, Optional
from enum import Enum


class PhilosopherState(Enum):
    THINKING = "thinking"
    REQUESTING = "requesting"
    EATING = "eating"


class ChandyMistraPhilosopher:
    """Philosopher in Chandy-Mistra solution"""
    
    def __init__(self, id: int, n_philosophers: int):
        self.id = id
        self.n_philosophers = n_philosophers
        self.state = PhilosopherState.THINKING
        self.eating_count = 0
        self.priority = random.uniform(0.1, 1.0)
        self.request_timestamp = 0
        self.eating_time = 0
        self.thinking_time = 0
        
        # Clean/dirty forks (True = clean, False = dirty)
        self.left_fork_clean = True
        self.right_fork_clean = True
        self.has_left_fork = False
        self.has_right_fork = False
    
    def want_to_eat(self) -> bool:
        """Check if philosopher wants to eat"""
        # Want to eat based on priority and thinking time
        return (self.state == PhilosopherState.THINKING and 
                self.thinking_time > self.priority * 10)
    
    def can_start_eating(self) -> bool:
        """Check if can start eating (has both clean forks)"""
        return (self.has_left_fork and self.has_right_fork and 
                self.left_fork_clean and self.right_fork_clean)


class ChandyMistraAlgorithm:
    """Chandy-Mistra solution using priorities and clean/dirty forks"""
    
    def __init__(self, n_philosophers: int = 5):
        self.n_philosophers = n_philosophers
        self.philosophers = [ChandyMistraPhilosopher(i, n_philosophers) for i in range(n_philosophers)]
        self.step_count = 0
        self.eating_durations = [0] * n_philosophers
        self.eating_time = 3
        self.global_timestamp = 0
        
        # Initialize fork distribution
        self._initialize_forks()
    
    def _initialize_forks(self):
        """Initialize fork distribution"""
        for i in range(self.n_philosophers):
            # Each philosopher starts with their left fork
            self.philosophers[i].has_left_fork = True
            self.philosophers[i].left_fork_clean = True
            
            # Right fork belongs to right neighbor initially
            right_neighbor = (i + 1) % self.n_philosophers
            self.philosophers[right_neighbor].has_right_fork = False
    
    def reset(self) -> Dict[str, Any]:
        """Reset to initial state"""
        self.philosophers = [ChandyMistraPhilosopher(i, self.n_philosophers) for i in range(self.n_philosophers)]
        self.step_count = 0
        self.eating_durations = [0] * self.n_philosophers
        self.global_timestamp = 0
        self._initialize_forks()
        return self.get_state()
    
    def handle_fork_requests(self):
        """Handle fork requests between philosophers"""
        for i, phil in enumerate(self.philosophers):
            if phil.state == PhilosopherState.REQUESTING:
                left_neighbor = (i - 1) % self.n_philosophers
                right_neighbor = (i + 1) % self.n_philosophers
                
                # Request left fork from left neighbor
                if not phil.has_left_fork:
                    self._request_fork(i, left_neighbor, 'right')
                
                # Request right fork from right neighbor  
                if not phil.has_right_fork:
                    self._request_fork(i, right_neighbor, 'left')
    
    def _request_fork(self, requester_id: int, holder_id: int, fork_side: str):
        """Request fork from neighbor"""
        requester = self.philosophers[requester_id]
        holder = self.philosophers[holder_id]
        
        # Determine which fork is being requested
        if fork_side == 'left':
            has_fork = holder.has_left_fork
            fork_clean = holder.left_fork_clean
        else:
            has_fork = holder.has_right_fork
            fork_clean = holder.right_fork_clean
        
        # Fork transfer conditions
        should_transfer = False
        
        if has_fork:
            if holder.state == PhilosopherState.THINKING:
                # Transfer if holder is thinking and fork is dirty
                should_transfer = not fork_clean
            elif holder.state == PhilosopherState.REQUESTING:
                # Transfer to higher priority requester
                should_transfer = requester.priority > holder.priority
        
        if should_transfer:
            self._transfer_fork(holder_id, requester_id, fork_side)
    
    def _transfer_fork(self, from_id: int, to_id: int, fork_side: str):
        """Transfer fork between philosophers"""
        from_phil = self.philosophers[from_id]
        to_phil = self.philosophers[to_id]
        
        if fork_side == 'left':
            # Transfer left fork
            from_phil.has_left_fork = False
            from_phil.left_fork_clean = False
            
            to_phil.has_left_fork = True
            to_phil.left_fork_clean = True
        else:
            # Transfer right fork
            from_phil.has_right_fork = False
            from_phil.right_fork_clean = False
            
            to_phil.has_right_fork = True
            to_phil.right_fork_clean = True
    
    def step(self) -> Dict[str, Any]:
        """Execute one step"""
        self.step_count += 1
        self.global_timestamp += 1
        
        # Update philosopher states
        for i, phil in enumerate(self.philosophers):
            if phil.state == PhilosopherState.THINKING:
                phil.thinking_time += 1
                if phil.want_to_eat():
                    phil.state = PhilosopherState.REQUESTING
                    phil.request_timestamp = self.global_timestamp
            
            elif phil.state == PhilosopherState.REQUESTING:
                if phil.can_start_eating():
                    phil.state = PhilosopherState.EATING
                    phil.eating_time = 0
                    phil.thinking_time = 0
            
            elif phil.state == PhilosopherState.EATING:
                self.eating_durations[i] += 1
                phil.eating_time += 1
                
                if self.eating_durations[i] >= self.eating_time:
                    # Finish eating
                    phil.state = PhilosopherState.THINKING
                    phil.eating_count += 1
                    phil.eating_time = 0
                    self.eating_durations[i] = 0
                    
                    # Make forks dirty after eating
                    phil.left_fork_clean = False
                    phil.right_fork_clean = False
        
        # Handle fork requests
        self.handle_fork_requests()
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'step': self.step_count,
            'philosopher_states': [p.state.value for p in self.philosophers],
            'eating_counts': [p.eating_count for p in self.philosophers],
            'priorities': [p.priority for p in self.philosophers],
            'currently_eating': sum(1 for p in self.philosophers if p.state == PhilosopherState.EATING),
            'currently_requesting': sum(1 for p in self.philosophers if p.state == PhilosopherState.REQUESTING),
            'deadlocked': self._detect_deadlock()
        }
    
    def _detect_deadlock(self) -> bool:
        """Simple deadlock detection"""
        requesting = [p for p in self.philosophers if p.state == PhilosopherState.REQUESTING]
        if len(requesting) >= self.n_philosophers - 1:
            # Most philosophers requesting - potential deadlock
            return True
        return False
    
    def run_simulation(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run complete simulation"""
        self.reset()
        states = []
        deadlock_count = 0
        
        for _ in range(max_steps):
            state = self.step()
            states.append(state)
            
            if state['deadlocked']:
                deadlock_count += 1
            
            # Stop if all philosophers have eaten multiple times
            if all(p.eating_count >= 3 for p in self.philosophers):
                break
        
        final_state = self.get_state()
        eating_counts = final_state['eating_counts']
        
        return {
            'algorithm': 'chandy-mistra',
            'final_eating_counts': eating_counts,
            'total_eating_events': sum(eating_counts),
            'eating_variance': np.var(eating_counts),
            'fairness_index': 1.0 / (1.0 + np.var(eating_counts)),
            'steps_taken': self.step_count,
            'deadlock_occurred': deadlock_count > 0,
            'deadlock_steps': deadlock_count,
            'states_history': states
        }