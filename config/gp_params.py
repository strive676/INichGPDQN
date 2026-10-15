"""Default parameters for genetic programming and rule evolution."""
from dataclasses import dataclass

@dataclass(frozen=True)
class GPParams:
    pop_size: int
    n_elites: int
    max_gen: int
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    n_immigrants: int = 20
    min_init_depth: int = 2
    max_init_depth: int = 6
    max_tree_depth: int = 8
    n_sampled_points: int = 40
    n_urgent: int = 15
    n_load: int = 15
    n_energy: int = 10
    tournament_size: int = 3



    elite_mode: str = "xu_clearing"
    quality_gap: float = 0.5
    clearing_delta: float = 0.0
    min_k: int = 2

SMOKE_GP = GPParams(pop_size=20, n_elites=10, max_gen=5, n_immigrants=2)
FULL_GP = GPParams(pop_size=200, n_elites=10, max_gen=50, n_immigrants=20)
