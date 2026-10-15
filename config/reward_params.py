"""Configuration parameters for scheduling reward functions."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RewardParams:
    beta: float = 50.0
    w_s: float = 0.5
    w_e: float = 0.5
