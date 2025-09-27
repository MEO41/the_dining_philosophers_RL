# Classical vs RL Algorithms Comparison Report
Generated at: 2025-09-27 12:37:09

## Summary Statistics

**Best Eating Performance:** dijkstra (Classical) - 568.00 events
**Best Fairness:** chandy-mistra (Classical) - 1.000
**Fastest Completion:** waiter (Classical) - 28 steps

## Detailed Results

### Classical Algorithms
**Dijkstra:**
- Eating Events: 568.00 ± 0.00
- Fairness: 0.000 ± 0.000
- Steps: 1000 ± 0
- Deadlock Rate: 0.0%

**Waiter:**
- Eating Events: 15.35 ± 0.65
- Fairness: 0.960 ± 0.071
- Steps: 28 ± 2
- Deadlock Rate: 0.0%

**Chandy-Mistra:**
- Eating Events: 0.00 ± 0.00
- Fairness: 1.000 ± 0.000
- Steps: 1000 ± 0
- Deadlock Rate: 100.0%

### RL Algorithms
**PPO:**
- Eating Events: 0.20 ± 0.68
- Fairness: 0.964 ± 0.131
- Steps: 106 ± 49
- Mean Reward: -920.81 ± 519.96

**A2C:**
- Eating Events: 0.30 ± 0.56
- Fairness: 0.963 ± 0.065
- Steps: 1000 ± 0
- Mean Reward: -2384.16 ± 4.32
