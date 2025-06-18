from dataclasses import dataclass
from utils import load_liver_data_mat, compute_voxel_distance_matrix, apply_gamma_to_matrix
from scipy.sparse import csc_matrix
import numpy as np
from config import Parameters, Gamma_parameters


D, phi, voxel_positions, tumor_voxels, H_1_voxels, H_2_voxels = load_liver_data_mat("data/liverEx_2.mat")
voxel_distance_matrix = compute_voxel_distance_matrix(voxel_positions)
gamma_matrix = apply_gamma_to_matrix(voxel_distance_matrix, Gamma_parameters)

@dataclass
class OptimizationData:
    D: csc_matrix
    phi: np.ndarray
    voxel_positions: csc_matrix
    tumor_voxels: np.ndarray
    H_1_voxels: np.ndarray
    H_2_voxels: np.ndarray
    parameters: Parameters








