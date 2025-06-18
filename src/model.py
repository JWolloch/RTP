from preprocessor import Preprocessor
from config import OptimizationParameters
import utils

import gurobipy as gp
from gurobipy import GRB

import logging

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, preprocessor: Preprocessor, optimization_parameters: OptimizationParameters):
        self._preprocessor = preprocessor
        self._optimization_parameters = optimization_parameters

        self._D = self._preprocessor.D
        self._n = self._D.shape[1] # n is the number of beamlets
        #D is a matrix of shape (number_of_voxels, number_of_beamlets), number_of_voxels includes healthy organ as well as tumor voxels
        self._N = self._optimization_parameters.N # N is the number of fractions (this implementation supports only N=2)
        self._mu_F = self._optimization_parameters.mu_F # mu_F - fractional homogeneity parameter
        self._d_bar_F = self._optimization_parameters.d_bar_F # d_bar_F is the maximum fractional radiation dose

        self._model = gp.Model()

        self._x = self.initialize_beamlet_intensity_variables()
        self._d_underbar_F = self.initialize_minimum_fractional_dose_variable()
        self._d_underbar = self.initialize_minimum_total_dose_variable()

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
    
    def fractional_dose(self, f: int, v: int) -> gp.LinExpr:
        """
        Computes the fractional dose for a given fraction and voxel.

        Args:
            f (int): Fraction number (0 or 1)
            v (int): Voxel index

        Returns:
            gp.LinExpr: The fractional dose for the given fraction and voxel.
        """
        return gp.quicksum(self._D[v, i] * self._x[f, i] for i in range(self._n))

    
    def total_dose(self, v: int) -> gp.LinExpr:
        """
        Computes the total dose for a given voxel.

        Args:
            v (int): Voxel index

        Returns:
            gp.LinExpr: The total dose for the given voxel.
        """
        return gp.quicksum(self.fractional_dose(f, v) for f in range(self._N))