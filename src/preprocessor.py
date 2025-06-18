import utils
import numpy as np
from config import GammaParameters, ProjectionParameters
import logging
from tabulate import tabulate
from scipy.sparse import csc_matrix

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, filepath: str):
        self._D, self._phi_hat, self._voxel_positions, self._tumor_voxels, self._H_1_voxels, self._H_2_voxels = utils.load_liver_data_mat(filepath)
        self._voxel_distance_matrix = utils.compute_voxel_distance_matrix(self._voxel_positions)
        self._gamma_matrix = utils.apply_gamma_to_matrix(self._voxel_distance_matrix, GammaParameters)
        self._phi_underbar_0 = utils.compute_phi_underbar_0(self._phi_hat, ProjectionParameters.delta)
        self._phi_bar_0 = utils.compute_phi_bar_0(self._phi_hat, ProjectionParameters.delta)
        self._phi_underbar_1 = utils.compute_phi_underbar_1(self._phi_underbar_0, self._gamma_matrix)
        self._phi_bar_1 = utils.compute_phi_bar_1(self._phi_bar_0, self._gamma_matrix)
        self._phi_underbar_2 = utils.compute_phi_underbar_2(self._phi_underbar_1, ProjectionParameters.sigma)
        self._phi_bar_2 = utils.compute_phi_bar_2(self._phi_bar_1, ProjectionParameters.sigma)
        self._M_3c1_1, self._M_3c1_2 = utils.compute_constraint_3c_1_coefficient_matrix(self._phi_bar_1, self._phi_underbar_1, self._phi_bar_2, self._phi_underbar_2, self._gamma_matrix)
        self._M_3c2_1, self._M_3c2_2 = utils.compute_constraint_3c_2_coefficient_matrix(self._phi_bar_1, self._phi_underbar_1, self._phi_bar_2, self._phi_underbar_2, self._gamma_matrix)

    def check_phi_bounds(self):
        logger.test("Checking projection bounds...")
        assert np.all(self._phi_underbar_0 <= self._phi_bar_0), "phi_underbar_0 <= phi_bar_0"
        assert np.all(self._phi_underbar_1 <= self._phi_bar_1), "phi_underbar_1 <= phi_bar_1"
        assert np.all(self._phi_underbar_2 <= self._phi_bar_2), "phi_underbar_2 <= phi_bar_2"
        logger.test("Projection bounds checked successfully\n")
    
    def print_min_max_projections(self):
        logger.preprocess("Printing min and max projections...")
        table = [
            ["min{phi_hat}", np.min(self._phi_hat).round(3)],
            ["max{phi_hat}", np.max(self._phi_hat).round(3)],
            ["min{phi_underbar_0}", np.min(self._phi_underbar_0).round(3)],
            ["max{phi_underbar_0}", np.max(self._phi_underbar_0).round(3)],
            ["min{phi_bar_0}", np.min(self._phi_bar_0).round(3)],
            ["max{phi_bar_0}", np.max(self._phi_bar_0).round(3)],
            ["min{phi_underbar_1}", np.min(self._phi_underbar_1).round(3)],
            ["max{phi_underbar_1}", np.max(self._phi_underbar_1).round(3)],
            ["min{phi_bar_1}", np.min(self._phi_bar_1).round(3)],
            ["max{phi_bar_1}", np.max(self._phi_bar_1).round(3)],
            ["min{phi_underbar_2}", np.min(self._phi_underbar_2).round(3)],
            ["max{phi_underbar_2}", np.max(self._phi_underbar_2).round(3)],
            ["min{phi_bar_2}", np.min(self._phi_bar_2).round(3)],
            ["max{phi_bar_2}", np.max(self._phi_bar_2).round(3)],
        ]
        print(tabulate(table, headers=["Projection", "Value"], tablefmt="grid"))
        print()
    
    def print_sample_projections(self):
        logger.preprocess("Printing projections on a random voxel...")
        v = np.random.randint(0, self._phi_hat.shape[0])
        table = [
            [f"phi_hat[{v}]", self._phi_hat[v].round(3)],
            [f"phi_underbar_0[{v}]", self._phi_underbar_0[v].round(3)],
            [f"phi_bar_0[{v}]", self._phi_bar_0[v].round(3)],
            [f"phi_underbar_1[{v}]", self._phi_underbar_1[v].round(3)],
            [f"phi_bar_1[{v}]", self._phi_bar_1[v].round(3)],
            [f"phi_underbar_2[{v}]", self._phi_underbar_2[v].round(3)],
            [f"phi_bar_2[{v}]", self._phi_bar_2[v].round(3)],
        ]
        print(tabulate(table, headers=["Projection", "Value"], tablefmt="grid"))
        print()
    
    @property
    def D(self) -> csc_matrix:
        return self._D
    
    @property
    def phi_hat(self) -> np.ndarray:
        return self._phi_hat
    
    @property   
    def voxel_positions(self) -> csc_matrix:
        return self._voxel_positions
    
    @property
    def tumor_voxels(self) -> np.ndarray:
        return self._tumor_voxels
    
    @property
    def H_1_voxels(self) -> np.ndarray:
        return self._H_1_voxels
    
    @property
    def H_2_voxels(self) -> np.ndarray:
        return self._H_2_voxels
    
    @property
    def voxel_distance_matrix(self) -> np.ndarray:
        return self._voxel_distance_matrix
    
    @property
    def gamma_matrix(self) -> np.ndarray:
        return self._gamma_matrix
    
    @property
    def phi_underbar_0(self) -> np.ndarray:
        return self._phi_underbar_0
    
    @property
    def phi_bar_0(self) -> np.ndarray:
        return self._phi_bar_0
    
    @property
    def phi_underbar_1(self) -> np.ndarray:
        return self._phi_underbar_1
    
    @property
    def phi_bar_1(self) -> np.ndarray:
        return self._phi_bar_1
    
    @property
    def phi_underbar_2(self) -> np.ndarray:
        return self._phi_underbar_2
    
    @property
    def phi_bar_2(self) -> np.ndarray:
        return self._phi_bar_2
    
    @property
    def M_3c1_1(self) -> csc_matrix:
        """M_3c1_1 is the coefficient matrix for fraction 1 of constraint 3c1"""
        return self._M_3c1_1
    
    @property
    def M_3c1_2(self) -> csc_matrix:
        """M_3c1_2 is the coefficient matrix for fraction 2 of constraint 3c1"""
        return self._M_3c1_2
    
    @property
    def M_3c2_1(self) -> csc_matrix:
        """M_3c2_1 is the coefficient matrix for fraction 1 of constraint 3c2"""
        return self._M_3c2_1
    
    @property
    def M_3c2_2(self) -> csc_matrix:
        """M_3c2_2 is the coefficient matrix for fraction 2 of constraint 3c2"""
        return self._M_3c2_2