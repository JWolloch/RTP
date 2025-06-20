from dataclasses import dataclass

@dataclass
class GammaParameters:
    max_dist: float = 100.0
    gamma_constant: float = 0.04
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
    debug: bool = True
    debug_n: int = 100
    N: int = 2
    lam: float = 0.5
    mu_F: float = 2
    d_bar_F: float = 0.95
    d_bar: float = 0.9 * d_bar_F * N
    d_bar_F_organ_1: float = 0.95
    d_bar_organ_1: float = 0.9 * d_bar_F_organ_1 * N
    d_bar_F_organ_2: float = 0.95
    d_bar_organ_2: float = 0.9 * d_bar_F_organ_2 * N