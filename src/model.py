from preprocessor import Preprocessor
from config import OptimizationParameters
import utils

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csc_matrix, hstack, eye

import logging

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, preprocessor: Preprocessor, optimization_parameters: OptimizationParameters):
        self._preprocessor = preprocessor
        self._optimization_parameters = optimization_parameters

        self._D = self._preprocessor.D # D is a matrix of shape (number_of_voxels, number_of_beamlets), number_of_voxels includes healthy organ as well as tumor voxels
        self._m = self._D.shape[0] # m is the total number of voxels
        self._n = self._D.shape[1] # n is the number of beamlets
        self._T = self._preprocessor.phi_hat.shape[0] # T is the number of tumor voxels
        self._H_1 = self._preprocessor.H_1_voxels.shape[0] # H_1 is the number of voxels in organ 1
        self._H_2 = self._preprocessor.H_2_voxels.shape[0] # H_2 is the number of voxels in organ 2
        self._N = self._optimization_parameters.N # N is the number of fractions (this implementation supports only N=2)
        self._mu_F = self._optimization_parameters.mu_F # mu_F - fractional homogeneity parameter
        self._d_bar_F = self._optimization_parameters.d_bar_F # d_bar_F is the maximum fractional radiation dose

        self._model = gp.Model()

        self._x = self.initialize_beamlet_intensity_variables()
        self._d_underbar_F = self.initialize_minimum_fractional_dose_variable()
        self._d_underbar = self.initialize_minimum_total_dose_variable()

        self._dose_tumor_voxels, self._dose_healthy_voxels_organ_1, self._dose_healthy_voxels_organ_2 = self.initialize_fractional_dose_variables()

    def initialize_beamlet_intensity_variables(self):
        x = self._model.addMVar(shape=(self._N, self._n), name="x")
        logger.model(f"Initialized {self._N}x{self._n} beamlet intensity variables")
        return x
    
    def initialize_minimum_fractional_dose_variable(self):
        d_underbar_F = self._model.addVar(name="d_underbar_F")
        return d_underbar_F
    
    def initialize_minimum_total_dose_variable(self):
        d_underbar = self._model.addVar(name="d_underbar")
        return d_underbar
    
    def initialize_fractional_dose_variables(self):
        dose_tumor_voxels = self._model.addMVar(shape=(self._N, self._T), name="fractional_dose_tumor_voxels")
        dose_healthy_voxels_organ_1 = self._model.addMVar(shape=(self._N, self._H_1), name="fractional_dose_healthy_voxels_organ_1")
        dose_healthy_voxels_organ_2 = self._model.addMVar(shape=(self._N, self._H_2), name="fractional_dose_healthy_voxels_organ_2")
        logger.model(f"Initialized {self._N}x{self._T} fractional dose auxiliary variables for tumor voxels and {self._N}x{self._H_1} fractional dose auxiliary variables for healthy voxels in organ 1 and {self._N}x{self._H_2} fractional dose auxiliary variables for healthy voxels in organ 2")
        return dose_tumor_voxels, dose_healthy_voxels_organ_1, dose_healthy_voxels_organ_2
    
    def fractional_dose_constraint(self) -> None:
        """
        Initializes fractional dose constraints.
        """
        #We start with tumor voxels
        D_tumor_sparse = csc_matrix(self._D[:self._T])
        A_tumor = -1 * eye(self._T)
        blocks = [A_tumor, D_tumor_sparse]
        A = hstack(blocks, format="csc")

        tumor_var_list_1 = self._dose_tumor_voxels[0].tolist() + self._x[0].tolist()
        y_tumor_1 = gp.MVar.fromlist(tumor_var_list_1)
        tumor_var_list_2 = self._dose_tumor_voxels[1].tolist() + self._x[1].tolist()
        y_tumor_2 = gp.MVar.fromlist(tumor_var_list_2)
        
        self._model.addMConstr(A, y_tumor_1, GRB.EQUAL, np.zeros(self._T))
        self._model.addMConstr(A, y_tumor_2, GRB.EQUAL, np.zeros(self._T))
        
        #Now we do the same for healthy voxels in organ 1
        D_healthy_organ_1_sparse = csc_matrix(self._D[self._T:self._T + self._H_1])
        A_healthy_organ_1 = -1 * eye(self._H_1)
        blocks = [A_healthy_organ_1, D_healthy_organ_1_sparse]
        A = hstack(blocks, format="csc")

        healthy_organ_1_var_list_1 = self._dose_healthy_voxels_organ_1[0].tolist() + self._x[0].tolist()
        y_healthy_organ_1_1 = gp.MVar.fromlist(healthy_organ_1_var_list_1)
        healthy_organ_1_var_list_2 = self._dose_healthy_voxels_organ_1[1].tolist() + self._x[1].tolist()
        y_healthy_organ_1_2 = gp.MVar.fromlist(healthy_organ_1_var_list_2)

        self._model.addMConstr(A, y_healthy_organ_1_1, GRB.EQUAL, np.zeros(self._H_1))
        self._model.addMConstr(A, y_healthy_organ_1_2, GRB.EQUAL, np.zeros(self._H_1))

        #Now we do the same for healthy voxels in organ 2
        D_healthy_organ_2_sparse = csc_matrix(self._D[self._T + self._H_1:self._T + self._H_1 + self._H_2])
        A_healthy_organ_2 = -1 * eye(self._H_2)
        blocks = [A_healthy_organ_2, D_healthy_organ_2_sparse]
        A = hstack(blocks, format="csc")

        healthy_organ_2_var_list_1 = self._dose_healthy_voxels_organ_2[0].tolist() + self._x[0].tolist()
        y_healthy_organ_2_1 = gp.MVar.fromlist(healthy_organ_2_var_list_1)
        healthy_organ_2_var_list_2 = self._dose_healthy_voxels_organ_2[1].tolist() + self._x[1].tolist()
        y_healthy_organ_2_2 = gp.MVar.fromlist(healthy_organ_2_var_list_2)

        self._model.addMConstr(A, y_healthy_organ_2_1, GRB.EQUAL, np.zeros(self._H_2))
        self._model.addMConstr(A, y_healthy_organ_2_2, GRB.EQUAL, np.zeros(self._H_2))
    
    def initialize_constraint_3b(self):
        """
        Initializes the constraint 3b.
        """
        A1 = -1 * csc_matrix(np.ones((self._T, 1)))
        A2 = csc_matrix(np.diag(self._preprocessor.phi_underbar_1))
        blocks = [A1, A2]
        A = hstack(blocks, format="csc")

        var_list_1 = [self._d_underbar_F] + self._dose_tumor_voxels[0].tolist()
        y_1 = gp.MVar.fromlist(var_list_1)

        var_list_2 = [self._d_underbar_F] + self._dose_tumor_voxels[1].tolist()
        y_2 = gp.MVar.fromlist(var_list_2)

        self._model.addMConstr(A, y_1, GRB.GREATER_EQUAL, np.zeros(self._T))
        self._model.addMConstr(A, y_2, GRB.GREATER_EQUAL, np.zeros(self._T))