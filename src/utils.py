from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.csgraph import shortest_path
import numpy as np
from config import Gamma_parameters

def load_liver_data_mat(filepath: str) -> tuple[csc_matrix, np.ndarray, csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    mat_data = loadmat(filepath, squeeze_me=True)  # squeeze to remove singleton dimensions
    
    def maybe_sparse(x):
        return csc_matrix(x) if not issparse(x) else x

    D = maybe_sparse(mat_data["Dij"])
    phi = np.array(mat_data["omf_Vec"]).flatten()
    voxel_positions = maybe_sparse(mat_data["neighbors_Mat"])
    
    # V is a list of arrays, each array contains voxel indices for a group
    V = mat_data["V"]
    tumor_voxels = V[0].astype(int) - 1  # MATLAB is 1-based; convert to 0-based
    H_1_voxels = V[1].astype(int) - 1
    H_2_voxels = V[2].astype(int) - 1

    return D, phi, voxel_positions, tumor_voxels, H_1_voxels, H_2_voxels

def apply_gamma_to_matrix(matrix: np.ndarray, parameters: Gamma_parameters) -> np.ndarray:
    return np.vectorize(lambda x: gamma(x, parameters))(matrix)

def compute_voxel_distance_matrix(adj_matrix: csc_matrix) -> np.ndarray:
    """
    Computes the (unweighted) shortest path distance matrix
    for the graph corresponding to the adjacency matrix, using SciPy's csgraph utilities.
    """
    distances = shortest_path(csgraph=adj_matrix, directed=False, unweighted=True)
    return distances.astype(int)

def gamma(x: float, parameters: Gamma_parameters) -> float:
    x = max(x, 1e-6) #avoid log(0)
    if 1 <= x <= parameters.max_dist:
        return parameters.gamma_constant + parameters.alpha_0 + parameters.alpha_1 * x + parameters.alpha_2 * np.log(x)
    elif x > parameters.max_dist:
        return parameters.gamma_constant + parameters.alpha_0 + parameters.alpha_1 * parameters.max_dist + parameters.alpha_2 * np.log(parameters.max_dist)
    else:
        return 0.0









