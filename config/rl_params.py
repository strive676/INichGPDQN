"""Configuration parameters for reinforcement learning agents."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RLParams:
    state_dim: int = 22
    hidden_dim: int = 128
    gamma: float = 0.95
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 5000
    target_update_steps: int = 500
    n_episodes: int = 64
    n_seeds: int = 3
    grad_clip: float = 5.0
    eps_min: float = 0.1
    eps_decay_frac: float = 0.8


    gamma_horizon_frac: float = 0.0

SMOKE_RL = RLParams(n_episodes=8, n_seeds=1)
FULL_RL = RLParams(n_episodes=64, n_seeds=3)
