from dataclasses import dataclass

@dataclass
class Parameters:
    N: int = 2
    mu_F: float = 1.5
    d_bar_F: float = 0.95
    d_bar: float = 0.9 * d_bar_F * N
    delta: float = 0.05
    sigma: float = 0.05
    eps: float = 1e-6

@dataclass
class Gamma_parameters:
    max_dist: float = 100.0
    gamma_constant: float = 0.04
    alpha_0: float = 0.0292761
    alpha_1: float = -0.0013514
    alpha_2: float = 0.0128265