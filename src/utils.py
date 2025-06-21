import os
import psutil
import threading
import time
import json
from dataclasses import asdict
from typing import Any
from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse, vstack
from scipy.sparse.csgraph import shortest_path
import numpy as np
from config import GammaParameters, ProjectionParameters, OptimizationParameters
import logging

logger = logging.getLogger(__name__)

def load_liver_data_mat(filepath: str) -> tuple[csc_matrix, np.ndarray, csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    logger.preprocess("Loading liver data from %s", filepath)
    mat_data = loadmat(filepath, squeeze_me=True)  # squeeze to remove singleton dimensions
    
    def maybe_sparse(x):
        return csc_matrix(x) if not issparse(x) else x

    D = maybe_sparse(mat_data["Dij"])
    phi_hat = np.array(mat_data["omf_Vec"]).flatten()
    voxel_positions = maybe_sparse(mat_data["neighbors_Mat"])
    
    # V is a list of arrays, each array contains voxel indices for a group
    V = mat_data["V"]
    tumor_voxels = V[0].astype(int) - 1  # MATLAB is 1-based; convert to 0-based
    H_1_voxels = V[1].astype(int) - 1
    H_2_voxels = V[2].astype(int) - 1

    #get dose matrix only for tumor, organ 1 and organ 2 voxels
    D_tumor = D[tumor_voxels]
    D_H_1 = D[H_1_voxels]
    D_H_2 = D[H_2_voxels]


    D_hat = vstack([D_tumor, D_H_1, D_H_2], format="csc")


    #print row indices that have only zero entries
    nonzero_counts = np.diff(D_hat.indptr)
    zero_rows = np.where(nonzero_counts == 0)[0]

    for row in zero_rows:
        print(f"Row {row} has only zero entries")


    logger.preprocess("Liver data loaded successfully\n")


    logger.preprocess(f"H_1_voxels.shape: {H_1_voxels.shape}")
    logger.preprocess(f"H_2_voxels.shape: {H_2_voxels.shape}")
    logger.preprocess(f"D_hat.shape (Use only relevant entries from D): {D_hat.shape}")
    logger.preprocess(f"phi_hat.shape: {phi_hat.shape}")
    logger.preprocess(f"voxel_positions.shape: {voxel_positions.shape}")

    return D_hat, phi_hat, voxel_positions, tumor_voxels, H_1_voxels, H_2_voxels

def compute_voxel_distance_matrix(adj_matrix: csc_matrix) -> np.ndarray:
    """
    Computes the (unweighted) shortest path distance matrix
    for the graph corresponding to the adjacency matrix, using SciPy's csgraph utilities.
    """
    logger.preprocess("Computing voxel distance matrix...")
    distances = shortest_path(csgraph=adj_matrix, directed=False, unweighted=True)
    logger.preprocess("Voxel distance matrix computed successfully\n")
    return distances.astype(int)

def gamma(x: float, parameters: GammaParameters) -> float:
    x = max(x, 1e-6) #avoid log(0)
    if 1 <= x <= parameters.max_dist:
        return parameters.gamma_constant + parameters.alpha_0 + parameters.alpha_1 * x + parameters.alpha_2 * np.log(x)
    elif x > parameters.max_dist:
        return parameters.gamma_constant + parameters.alpha_0 + parameters.alpha_1 * parameters.max_dist + parameters.alpha_2 * np.log(parameters.max_dist)
    else:
        return 0.0
    
def apply_gamma_to_matrix(distances: np.ndarray, parameters: GammaParameters) -> np.ndarray:
    """
    Apply the gamma function elementwise to a matrix of distances
    using vectorized NumPy operations for speed.
    
    Args:
        distances (np.ndarray): Distance matrix (e.g. voxel distances)
        p (GammaParameters): Parameters with gamma_constant, alpha_0, alpha_1, alpha_2, max_dist

    Returns:
        np.ndarray: Matrix with gamma(x) applied elementwise
    """
    logger.preprocess("Applying gamma function to distance matrix...")
    x = np.maximum(distances, 1e-6)  # Avoid log(0)

    result = np.zeros_like(x, dtype=np.float64)

    # Conditions
    in_range = (x >= 1) & (x <= parameters.max_dist)
    above_range = x > parameters.max_dist

    # gamma(x) where 1 <= x <= max_dist
    result[in_range] = (
        parameters.gamma_constant +
        parameters.alpha_0 +
        parameters.alpha_1 * x[in_range] +
        parameters.alpha_2 * np.log(x[in_range])
    )

    # gamma(x) where x > max_dist (constant value)
    gamma_max = (
        parameters.gamma_constant +
        parameters.alpha_0 +
        parameters.alpha_1 * parameters.max_dist +
        parameters.alpha_2 * np.log(parameters.max_dist)
    )
    result[above_range] = gamma_max

    # For x < 1, result stays at 0
    logger.preprocess("Gamma function applied to distance matrix successfully\n")
    return result

def compute_phi_underbar_0(phi_hat: np.ndarray, delta: float) -> np.ndarray:
    logger.preprocess("Computing phi_underbar_0...")
    phi_underbar_0 = np.maximum(phi_hat - delta, 0)
    logger.preprocess("phi_underbar_0 computed successfully\n")
    return phi_underbar_0

def compute_phi_bar_0(phi_hat: np.ndarray, delta: float) -> np.ndarray:
    logger.preprocess("Computing phi_bar_0...")
    phi_bar_0 = np.minimum(phi_hat + delta, 1)
    logger.preprocess("phi_bar_0 computed successfully\n")
    return phi_bar_0

def compute_phi_underbar_1(phi_underbar_0: np.ndarray, gamma_matrix: np.ndarray) -> np.ndarray:
    logger.preprocess("Computing phi_underbar_1...")
    phi_underbar_1 = np.max(phi_underbar_0[:, None] - gamma_matrix, axis=0)
    logger.preprocess("phi_underbar_1 computed successfully\n")
    return phi_underbar_1

def compute_phi_bar_1(phi_bar_0: np.ndarray, gamma_matrix: np.ndarray) -> np.ndarray:
    logger.preprocess("Computing phi_bar_1...")
    phi_bar_1 = np.min(phi_bar_0[:, None] + gamma_matrix, axis=0)
    logger.preprocess("phi_bar_1 computed successfully\n")
    return phi_bar_1

def compute_phi_underbar_2(phi_underbar_1: np.ndarray, sigma: float) -> np.ndarray:
    logger.preprocess("Computing phi_underbar_2...")
    phi_underbar_2 = np.maximum(phi_underbar_1 - sigma, 0)
    logger.preprocess("phi_underbar_2 computed successfully\n")
    return phi_underbar_2

def compute_phi_bar_2(phi_bar_1: np.ndarray, sigma: float) -> np.ndarray:
    logger.preprocess("Computing phi_bar_2...")
    phi_bar_2 = np.minimum(phi_bar_1 + sigma, 1)
    logger.preprocess("phi_bar_2 computed successfully\n")
    return phi_bar_2

def compute_constraint_3c_1_coefficient_matrix(phi_bar_1: np.ndarray, phi_underbar_1: np.ndarray,
                                               phi_bar_2: np.ndarray, phi_underbar_2: np.ndarray,
                                               gamma_matrix: np.ndarray) -> tuple[csc_matrix, csc_matrix]:
    """
    Computes the coefficient matrices for constraint 3c1.
    """
    logger.preprocess("Computing constraint 3c 1 coefficient matrices...")
    M_1 = csc_matrix(np.maximum(phi_underbar_1[:, None], phi_bar_1[None, :] - gamma_matrix))
    M_2 = csc_matrix(np.maximum(phi_underbar_2[:, None], phi_bar_2[None, :] - gamma_matrix))

    assert M_1.shape == gamma_matrix.shape, "M_1 must have the same shape as gamma_matrix"
    assert M_2.shape == gamma_matrix.shape, "M_2 must have the same shape as gamma_matrix"

    logger.test("Constraint 3c 1 coefficient matrices computed successfully\n")
    return M_1, M_2

def compute_constraint_3c_2_coefficient_matrix(phi_bar_1: np.ndarray, phi_underbar_1: np.ndarray,
                                               phi_bar_2: np.ndarray, phi_underbar_2: np.ndarray,
                                               gamma_matrix: np.ndarray) -> tuple[csc_matrix, csc_matrix]:
    """
    Computes the coefficient matrices for constraint 3c2.
    """
    logger.preprocess("Computing constraint 3c 2 coefficient matrices...")
    M_1 = csc_matrix(np.maximum(phi_underbar_1[:, None] + gamma_matrix, phi_bar_1[None, :]))
    M_2 = csc_matrix(np.maximum(phi_underbar_2[:, None] + gamma_matrix, phi_bar_2[None, :]))

    assert M_1.shape == gamma_matrix.shape, "M_1 must have the same shape as gamma_matrix"
    assert M_2.shape == gamma_matrix.shape, "M_2 must have the same shape as gamma_matrix"

    logger.test("Constraint 3c 2 coefficient matrices computed successfully\n")
    return M_1, M_2

def save_run_results(
    gamma_params: GammaParameters,
    proj_params: ProjectionParameters,
    opt_params: OptimizationParameters,
    solve_time_seconds: float,
    peak_memory_mb: float,
    solution_dict: dict[str, Any],
):
    # Folder name based on solution method and constraint count
    folder_name = f"{opt_params.solution_method.name}_{opt_params.n_most_violated_constraints}"
    folder_path = os.path.join("results", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save parameters and solver time
    run_info = {
        "gamma_parameters": asdict(gamma_params),
        "projection_parameters": asdict(proj_params),
        "optimization_parameters": asdict(opt_params),
        "solve_time_seconds": round(solve_time_seconds, 3),
        "peak_memory_mb": round(peak_memory_mb, 3)
    }
    with open(os.path.join(folder_path, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=4)

    # Save solution dictionary
    with open(os.path.join(folder_path, "solution.json"), "w") as f:
        json.dump(solution_dict, f, indent=4)

class MemoryMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_memory = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor)

    def _monitor(self):
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            mem = process.memory_info().rss
            if mem > self.peak_memory:
                self.peak_memory = mem
            time.sleep(self.interval)

    def start(self):
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()