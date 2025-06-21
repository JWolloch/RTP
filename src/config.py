from dataclasses import dataclass
from enum import IntEnum

class SolutionMethod(IntEnum):
    PRIMAL_SIMPLEX = 0
    PRIMAL_DUAL_SIMPLEX = 1
    BARRIER = 2

@dataclass
class GammaParameters:
    max_dist: float = 10.0
    gamma_constant: float = 0.15
    alpha_0: float = 0.0292761
    alpha_1: float = -0.0013514
    alpha_2: float = 0.0128265

@dataclass
class ProjectionParameters:
    delta: float = 0.05
    sigma: float = 0.05
    eps: float = 1e-6

@dataclass
class OptimizationParameters:
    debug: bool = False
    debug_n: int = 1000
    solution_method: SolutionMethod = SolutionMethod.PRIMAL_SIMPLEX # Run with [Primal-simplex, Primal-dual-simplex, Barrier]
    row_generation: bool = True
    n_most_violated_constraints: int = 10 # Run with [2, 3, 5, 10]
    max_row_generation_iterations: int = 100
    N: int = 2
    lam: float = 0.5
    mu_F: float = 1.3
    d_bar_F: float = 0.95
    d_bar: float = 0.9 * d_bar_F * N
    d_bar_F_organ_1: float = 0.95
    d_bar_organ_1: float = 0.9 * d_bar_F_organ_1 * N
    d_bar_F_organ_2: float = 0.95
    d_bar_organ_2: float = 0.9 * d_bar_F_organ_2 * N