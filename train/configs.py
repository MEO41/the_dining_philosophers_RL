"""
Configuration file for training dining philosophers RL models
"""

# PPO Configuration (for continuous environment)
PPO_CONFIG = {
    'algorithm': 'ppo',
    'env_type': 'continuous',
    'n_philosophers': 5,
    'max_steps': 1000,
    
    # Network architecture
    'hidden_dim': 256,
    
    # Learning rates
    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    
    # PPO hyperparameters
    'gamma': 0.99,
    'epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    
    # Training parameters
    'n_episodes': 2000,
    'update_frequency': 20,  # Update every N episodes
    'save_frequency': 100,   # Save model every N episodes
    
    # Evaluation
    'eval_frequency': 50,    # Evaluate every N episodes
    'eval_episodes': 10,     # Number of episodes for evaluation
    
    # Logging
    'log_frequency': 10,     # Log stats every N episodes
}

# A2C Configuration (for discrete environment)
A2C_CONFIG = {
    'algorithm': 'a2c',
    'env_type': 'discrete',
    'n_philosophers': 5,
    'max_steps': 1000,
    
    # Network architecture
    'hidden_dim': 256,
    
    # Learning rates
    'actor_lr': 1e-3,
    'critic_lr': 1e-3,
    
    # A2C hyperparameters
    'gamma': 0.99,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    
    # Training parameters
    'n_episodes': 2000,
    'update_frequency': 1,   # Update every episode for A2C
    'save_frequency': 100,   # Save model every N episodes
    
    # Evaluation
    'eval_frequency': 50,    # Evaluate every N episodes
    'eval_episodes': 10,     # Number of episodes for evaluation
    
    # Logging
    'log_frequency': 10,     # Log stats every N episodes
}

# SAC Configuration (for continuous environment)
SAC_CONFIG = {
    'algorithm': 'sac',
    'env_type': 'continuous',
    'n_philosophers': 5,
    'max_steps': 1000,
    
    # Network architecture
    'hidden_dim': 256,
    
    # Learning rates
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'alpha_lr': 3e-4,
    
    # SAC hyperparameters
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'buffer_size': 1000000,
    'learning_starts': 10000,
    
    # Training parameters
    'n_episodes': 2000,
    'update_frequency': 1,   # Update every step
    'save_frequency': 100,
    
    # Evaluation
    'eval_frequency': 50,
    'eval_episodes': 10,
    
    # Logging
    'log_frequency': 10,
}

# DQN Configuration (for discrete environment)
DQN_CONFIG = {
    'algorithm': 'dqn',
    'env_type': 'discrete',
    'n_philosophers': 5,
    'max_steps': 1000,
    
    # Network architecture
    'hidden_dim': 256,
    
    # Learning parameters
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'batch_size': 64,
    'buffer_size': 100000,
    'target_update_freq': 1000,
    
    # Exploration
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    
    # Training parameters
    'n_episodes': 2000,
    'update_frequency': 1,
    'save_frequency': 100,
    
    # Evaluation
    'eval_frequency': 50,
    'eval_episodes': 10,
    
    # Logging
    'log_frequency': 10,
}
ENVIRONMENT_CONFIGS = {
    'discrete': {
        'name': 'DiscreteDiningPhilosophersEnv',
        'action_space_type': 'discrete',
        'action_dim': 6,  # 6 discrete actions per philosopher
    },
    'continuous': {
        'name': 'ContinuousDiningPhilosophersEnv',
        'action_space_type': 'continuous',
        'action_dim': 4,  # 4 continuous actions per philosopher
    }
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    'save_dir': 'models',
    'log_dir': 'logs',
    'results_dir': 'results',
    'tensorboard_dir': 'runs',
    
    # Metrics to track
    'metrics': [
        'episode_reward',
        'episode_length',
        'eating_counts',
        'eating_variance',
        'deadlock_events',
        'fairness_index',
        'actor_loss',
        'critic_loss',
        'entropy'
    ],
    
    # Early stopping
    'early_stopping': {
        'enabled': True,
        'patience': 200,
        'min_delta': 0.1,
        'metric': 'episode_reward'
    }
}

def get_config(algorithm: str):
    """Get configuration for specified algorithm"""
    configs = {
        'ppo': PPO_CONFIG,
        'a2c': A2C_CONFIG,
        'sac': SAC_CONFIG,
        'dqn': DQN_CONFIG
    }
    
    if algorithm.lower() in configs:
        return configs[algorithm.lower()]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def get_env_config(env_type: str):
    """Get environment configuration"""
    if env_type in ENVIRONMENT_CONFIGS:
        return ENVIRONMENT_CONFIGS[env_type]
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

# Hyperparameter search spaces (for future optimization)
HYPERPARAMETER_SPACES = {
    'ppo': {
        'actor_lr': [1e-4, 3e-4, 1e-3],
        'critic_lr': [3e-4, 1e-3, 3e-3],
        'epsilon': [0.1, 0.2, 0.3],
        'entropy_coef': [0.001, 0.01, 0.1],
        'hidden_dim': [128, 256, 512]
    },
    'a2c': {
        'actor_lr': [3e-4, 1e-3, 3e-3],
        'critic_lr': [3e-4, 1e-3, 3e-3],
        'entropy_coef': [0.001, 0.01, 0.1],
        'hidden_dim': [128, 256, 512]
    },
    'sac': {
        'actor_lr': [1e-4, 3e-4, 1e-3],
        'critic_lr': [1e-4, 3e-4, 1e-3],
        'alpha_lr': [1e-4, 3e-4, 1e-3],
        'tau': [0.001, 0.005, 0.01],
        'batch_size': [64, 128, 256]
    },
    'dqn': {
        'learning_rate': [1e-4, 1e-3, 3e-3],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'target_update_freq': [500, 1000, 2000],
        'hidden_dim': [128, 256, 512]
    }
}