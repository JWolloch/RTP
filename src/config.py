from dataclasses import dataclass
from enum import IntEnum
import numpy as np

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

@dataclass
class OptimizationParameters:
    debug: bool = False
    debug_n: int = 2000
    solution_method: SolutionMethod = SolutionMethod.PRIMAL_DUAL_SIMPLEX # Run with [Primal-simplex, Primal-dual-simplex, Barrier] barrier if it takes too long
    row_generation: bool = True
    n_most_violated_constraints: int = 10 # Run with [5, 10] priority 2
    max_constraint_addition: int = 10**10 #or 2000 priority 1
    max_row_generation_iterations: int = 100
    N: int = 2
    lam: float = 0.5
    mu_F: float = 1.3 # 1.1, 1.15, 1.2,... priority 3
    d_bar_organ_1: float = 60
    d_bar_F_organ_1: float = d_bar_organ_1/N * 1.1
    d_bar_organ_2: float = 50
    d_bar_F_organ_2: float = d_bar_organ_2/N * 1.1
    d_bar_organ_3: float = 60
    d_bar_F_organ_3: float = d_bar_organ_2/N * 1.1
    eps: float = 1e-4 # eps is the tolerance for the fractional dose constraints
