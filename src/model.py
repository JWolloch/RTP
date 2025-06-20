from preprocessor import Preprocessor
from config import OptimizationParameters
import utils

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csc_matrix, hstack, eye, diags

import logging

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, preprocessor: Preprocessor, optimization_parameters: OptimizationParameters, debug: bool=False):
        self._preprocessor = preprocessor
        self._optimization_parameters = optimization_parameters
        self._debug = debug
        if self._debug:
            self._D = self._preprocessor.D[:, :10]
        else:
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
        self._model.setParam(GRB.Param.DualReductions, 0)

        self._x = self.initialize_beamlet_intensity_variables()
        self._d_underbar_F = self.initialize_minimum_fractional_dose_variable()
        self._d_underbar = self.initialize_minimum_total_dose_variable()

        self._dose_tumor_voxels, self._dose_healthy_voxels_organ_1, self._dose_healthy_voxels_organ_2 = self.initialize_fractional_dose_variables()

        self._model_status = None

    def initialize_beamlet_intensity_variables(self):
        if self._debug:
            x = self._model.addMVar(shape=(self._N, 10), lb=0.01, name="x")
            logger.model(f"Initialized {self._N}x{10} beamlet intensity variables")
        else:
            x = self._model.addMVar(shape=(self._N, self._n), lb=0.01, name="x")
            logger.model(f"Initialized {self._N}x{self._n} beamlet intensity variables")
        return x
    
    def initialize_minimum_fractional_dose_variable(self):
        d_underbar_F = self._model.addVar(name="d_underbar_F")
        return d_underbar_F
    
    def initialize_minimum_total_dose_variable(self):
        d_underbar = self._model.addVar(name="d_underbar")
        return d_underbar
    
    def initialize_fractional_dose_variables(self):
        if self._debug:
            dose_tumor_voxels = self._model.addMVar(shape=(self._N, 10), name="fractional_dose_tumor_voxels")
            dose_healthy_voxels_organ_1 = self._model.addMVar(shape=(self._N, 10), name="fractional_dose_healthy_voxels_organ_1")
            dose_healthy_voxels_organ_2 = self._model.addMVar(shape=(self._N, 10), name="fractional_dose_healthy_voxels_organ_2")
            logger.model(f"Initialized {self._N}x{10} fractional dose auxiliary variables for tumor voxels and {self._N}x{10} fractional dose auxiliary variables for healthy voxels in organ 1 and {self._N}x{10} fractional dose auxiliary variables for healthy voxels in organ 2")
        else:
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
        if self._debug:
            I_tumor = -1 * eye(10)
            D_tumor_sparse = csc_matrix(self._D[:10])
            blocks = [I_tumor, D_tumor_sparse]
        else:
            I_tumor = -1 * eye(self._T)
            D_tumor_sparse = csc_matrix(self._D[:self._T])
            blocks = [I_tumor, D_tumor_sparse]

        A_tumor = hstack(blocks, format="csc")

        tumor_var_list_1 = self._dose_tumor_voxels[0].tolist() + self._x[0].tolist()
        y_tumor_1 = gp.MVar.fromlist(tumor_var_list_1)
        tumor_var_list_2 = self._dose_tumor_voxels[1].tolist() + self._x[1].tolist()
        y_tumor_2 = gp.MVar.fromlist(tumor_var_list_2)

        if self._debug:
            self._model.addMConstr(A_tumor, y_tumor_1, GRB.EQUAL, np.zeros(10), name="fractional_dose_constraint_tumor_1")
            self._model.addMConstr(A_tumor, y_tumor_2, GRB.EQUAL, np.zeros(10), name="fractional_dose_constraint_tumor_2")
        else:
            self._model.addMConstr(A_tumor, y_tumor_1, GRB.EQUAL, np.zeros(self._T), name="fractional_dose_constraint_tumor_1")
            self._model.addMConstr(A_tumor, y_tumor_2, GRB.EQUAL, np.zeros(self._T), name="fractional_dose_constraint_tumor_2")
        
        #Now we do the same for healthy voxels in organ 1
        if self._debug:
            I_healthy_organ_1 = -1 * eye(10)
            D_healthy_organ_1_sparse = csc_matrix(self._D[self._T:self._T + 10])
            blocks = [I_healthy_organ_1, D_healthy_organ_1_sparse]
        else:
            I_healthy_organ_1 = -1 * eye(self._H_1)
            D_healthy_organ_1_sparse = csc_matrix(self._D[self._T:self._T + self._H_1])
            blocks = [I_healthy_organ_1, D_healthy_organ_1_sparse]

        A_healthy_organ_1 = hstack(blocks, format="csc")

        healthy_organ_1_var_list_1 = self._dose_healthy_voxels_organ_1[0].tolist() + self._x[0].tolist()
        y_healthy_organ_1_1 = gp.MVar.fromlist(healthy_organ_1_var_list_1)
        healthy_organ_1_var_list_2 = self._dose_healthy_voxels_organ_1[1].tolist() + self._x[1].tolist()
        y_healthy_organ_1_2 = gp.MVar.fromlist(healthy_organ_1_var_list_2)

        if self._debug:
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_1, GRB.EQUAL, np.zeros(10), name="fractional_dose_constraint_healthy_organ_1_1")
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_2, GRB.EQUAL, np.zeros(10), name="fractional_dose_constraint_healthy_organ_1_2")
        else:
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_1, GRB.EQUAL, np.zeros(self._H_1), name="fractional_dose_constraint_healthy_organ_1_1")
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_2, GRB.EQUAL, np.zeros(self._H_1), name="fractional_dose_constraint_healthy_organ_1_2")

        #Now we do the same for healthy voxels in organ 2
        if self._debug:
            I_healthy_organ_2 = -1 * eye(10)
            D_healthy_organ_2_sparse = csc_matrix(self._D[self._T + self._H_1:self._T + self._H_1 + 10])
            blocks = [I_healthy_organ_2, D_healthy_organ_2_sparse]
        else:
            I_healthy_organ_2 = -1 * eye(self._H_2)
            D_healthy_organ_2_sparse = csc_matrix(self._D[self._T + self._H_1:self._T + self._H_1 + self._H_2])
            blocks = [I_healthy_organ_2, D_healthy_organ_2_sparse]

        A_healthy_organ_2 = hstack(blocks, format="csc")

        healthy_organ_2_var_list_1 = self._dose_healthy_voxels_organ_2[0].tolist() + self._x[0].tolist()
        y_healthy_organ_2_1 = gp.MVar.fromlist(healthy_organ_2_var_list_1)
        healthy_organ_2_var_list_2 = self._dose_healthy_voxels_organ_2[1].tolist() + self._x[1].tolist()
        y_healthy_organ_2_2 = gp.MVar.fromlist(healthy_organ_2_var_list_2)

        if self._debug:
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_1, GRB.EQUAL, np.zeros(10), name="fractional_dose_constraint_healthy_organ_2_1")
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_2, GRB.EQUAL, np.zeros(10), name="fractional_dose_constraint_healthy_organ_2_2")
        else:
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_1, GRB.EQUAL, np.zeros(self._H_2), name="fractional_dose_constraint_healthy_organ_2_1")
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_2, GRB.EQUAL, np.zeros(self._H_2), name="fractional_dose_constraint_healthy_organ_2_2")
    
    def initialize_constraint_3b(self):
        """
        Initializes the constraint 3b.
        """
        if self._debug:
            A1 = -1 * csc_matrix(np.ones((10, 1)))
            A2 = csc_matrix(np.diag(self._preprocessor.phi_underbar_1[:10]))
            blocks = [A1, A2]
        else:
            A1 = -1 * csc_matrix(np.ones((self._T, 1)))
            A2 = csc_matrix(np.diag(self._preprocessor.phi_underbar_1))
            blocks = [A1, A2]

        A = hstack(blocks, format="csc")

        var_list_1 = [self._d_underbar_F] + self._dose_tumor_voxels[0].tolist()
        y_1 = gp.MVar.fromlist(var_list_1)

        var_list_2 = [self._d_underbar_F] + self._dose_tumor_voxels[1].tolist()
        y_2 = gp.MVar.fromlist(var_list_2)

        if self._debug:
            self._model.addMConstr(A, y_1, GRB.GREATER_EQUAL, np.zeros(10), name="constraint_3b_1")
            self._model.addMConstr(A, y_2, GRB.GREATER_EQUAL, np.zeros(10), name="constraint_3b_2")
        else:
            self._model.addMConstr(A, y_1, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3b_1")
            self._model.addMConstr(A, y_2, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3b_2")
    
    def initialize_constraint_3c1(self):
        """
        Initializes the constraint 3c1.
        """
        if self._debug:
            tumor_vars_f1 = self._dose_tumor_voxels[0].tolist()
            tumor_vars_f2 = self._dose_tumor_voxels[1].tolist()
            logger.model(f"Building constraint 3c1 for {10} tumor voxels...")
        else:
            tumor_vars_f1 = self._dose_tumor_voxels[0].tolist()
            tumor_vars_f2 = self._dose_tumor_voxels[1].tolist()
            logger.model(f"Building constraint 3c1 for {self._T} tumor voxels...")
        
        if self._debug:
            for v in range(10):
                logger.model(f"Constraint 3c1 progress: {v}/{10} voxels processed")
                #======== Fraction 1 =========
                A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((10, 1)))
                A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[:, v][:10].toarray().flatten())

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                var_list_1 = [tumor_vars_f1[v]] + tumor_vars_f1
                y_1 = gp.MVar.fromlist(var_list_1)

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(10), name=f"constraint_3c1_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((10, 1)))
                B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[:, v][:10].toarray().flatten())

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                var_list_2 = [tumor_vars_f2[v]] + tumor_vars_f2
                y_2 = gp.MVar.fromlist(var_list_2)

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(10), name=f"constraint_3c1_2_{v}")
            logger.model("Constraint 3c1 completed.")

        else:

            for v in range(self._T):
                if v % 100 == 0:  # Log progress every 100 voxels
                    logger.model(f"Constraint 3c1 progress: {v}/{self._T} voxels processed")
                #======== Fraction 1 =========
                A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((self._T, 1)))
                A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[:, v].toarray().flatten())

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                var_list_1 = [tumor_vars_f1[v]] + tumor_vars_f1
                y_1 = gp.MVar.fromlist(var_list_1)

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c1_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((self._T, 1)))
                B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[:, v].toarray().flatten())

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                var_list_2 = [tumor_vars_f2[v]] + tumor_vars_f2
                y_2 = gp.MVar.fromlist(var_list_2)

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c1_2_{v}")
            logger.model("Constraint 3c1 completed.")
    
    def initialize_constraint_3c2(self):
        """
        Initializes the constraint 3c2.
        """
        tumor_vars_f1 = self._dose_tumor_voxels[0].tolist()
        tumor_vars_f2 = self._dose_tumor_voxels[1].tolist()
        
        if self._debug:
            logger.model(f"Building constraint 3c2 for {10} tumor voxels...")
        else:
            logger.model(f"Building constraint 3c2 for {self._T} tumor voxels...")

        if self._debug:
            for v in range(10):
                logger.model(f"Constraint 3c2 progress: {v}/{10} voxels processed")

                #======== Fraction 1 =========
                A1 = self._preprocessor.M_3c2_1[:, v][:10]
                A2 = -self._mu_F * diags(self._preprocessor.phi_bar_1[:10])

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                var_list_1 = [tumor_vars_f1[v]] + tumor_vars_f1
                y_1 = gp.MVar.fromlist(var_list_1)

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(10), name=f"constraint_3c2_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.M_3c2_2[:, v][:10]
                B2 = -self._mu_F * diags(self._preprocessor.phi_bar_2[:10])

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                var_list_2 = [tumor_vars_f2[v]] + tumor_vars_f2
                y_2 = gp.MVar.fromlist(var_list_2)

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(10), name=f"constraint_3c2_2_{v}")
            logger.model("Constraint 3c2 completed.")
        else:
            for v in range(self._T):
                if v % 100 == 0:  # Log progress every 100 voxels
                    logger.model(f"Constraint 3c2 progress: {v}/{self._T} voxels processed")
                #======== Fraction 1 =========
                A1 = self._preprocessor.M_3c2_1[:, v]
                A2 = -self._mu_F * diags(self._preprocessor.phi_bar_1)

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                var_list_1 = [tumor_vars_f1[v]] + tumor_vars_f1
                y_1 = gp.MVar.fromlist(var_list_1)

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c2_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.M_3c2_2[:, v]
                B2 = -self._mu_F * diags(self._preprocessor.phi_bar_2)

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                var_list_2 = [tumor_vars_f2[v]] + tumor_vars_f2
                y_2 = gp.MVar.fromlist(var_list_2)

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c2_2_{v}")
            logger.model("Constraint 3c2 completed.")
        
    def initialize_constraint_3d(self):
        """
        Initializes the constraint 3d.
        """
        if self._debug:
            A1 = -1 * csc_matrix(np.ones((10, 1)))
            A2 = diags(self._preprocessor.phi_underbar_1[:10])
            A3 = diags(self._preprocessor.phi_underbar_2[:10])
            blocks = [A1, A2, A3]
        else:
            A1 = -1 * csc_matrix(np.ones((self._T, 1)))
            A2 = diags(self._preprocessor.phi_underbar_1)
            A3 = diags(self._preprocessor.phi_underbar_2)
            blocks = [A1, A2, A3]
        A = hstack(blocks, format="csc")

        var_list = [self._d_underbar] + self._dose_tumor_voxels[0].tolist() + self._dose_tumor_voxels[1].tolist()
        y = gp.MVar.fromlist(var_list)

        if self._debug:
            self._model.addMConstr(A, y, GRB.GREATER_EQUAL, np.zeros(10), name="constraint_3d")
        else:
            self._model.addMConstr(A, y, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3d")
    
    def initialize_constraint_3e(self):
        """
        Initializes the constraint 3e.
        """
        if self._debug:
            A_organ_1 = eye(10)
            A_organ_2 = eye(10)
        else:
            A_organ_1 = eye(self._H_1)
            A_organ_2 = eye(self._H_2)

        #======== Organ 1 =========
        y_1 = self._dose_healthy_voxels_organ_1[0]
        y_2 = self._dose_healthy_voxels_organ_1[1]

        if self._debug:
            self._model.addMConstr(A_organ_1, y_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(10), name="constraint_3e_1")
            self._model.addMConstr(A_organ_1, y_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(10), name="constraint_3e_2")
        else:
            self._model.addMConstr(A_organ_1, y_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3e_1")
            self._model.addMConstr(A_organ_1, y_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3e_2")

        #======== Organ 2 =========
        z_1 = self._dose_healthy_voxels_organ_2[0]
        z_2 = self._dose_healthy_voxels_organ_2[1]

        if self._debug:
            self._model.addMConstr(A_organ_2, z_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(10), name="constraint_3e_3")
            self._model.addMConstr(A_organ_2, z_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(10), name="constraint_3e_4")
        else:
            self._model.addMConstr(A_organ_2, z_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3e_3")
            self._model.addMConstr(A_organ_2, z_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3e_4")
    
    def initialize_constraint_3f(self):
        """
        Initializes the constraint 3f.
        """
        if self._debug:
            I_organ_1 = eye(10)
            I_organ_2 = eye(10)
        else:
            I_organ_1 = eye(self._H_1)
            I_organ_2 = eye(self._H_2)

        A_organ_1 = hstack([I_organ_1, I_organ_1], format="csc")
        A_organ_2 = hstack([I_organ_2, I_organ_2], format="csc")

        #======== Organ 1 =========
        y = self._dose_healthy_voxels_organ_1[0].tolist() + self._dose_healthy_voxels_organ_1[1].tolist()

        if self._debug:
            self._model.addMConstr(A_organ_1, y, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_1 * np.ones(10), name="constraint_3f_1")
        else:
            self._model.addMConstr(A_organ_1, y, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_1 * np.ones(self._H_1), name="constraint_3f_1")

        #======== Organ 2 =========
        z = self._dose_healthy_voxels_organ_2[0].tolist() + self._dose_healthy_voxels_organ_2[1].tolist()

        if self._debug:
            self._model.addMConstr(A_organ_2, z, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_2 * np.ones(10), name="constraint_3f_2")
        else:
            self._model.addMConstr(A_organ_2, z, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_2 * np.ones(self._H_2), name="constraint_3f_2")
    
    def build_model(self) -> None:
        """
        Builds the model.
        """
        logger.model("Building model constraints...")
        logger.model("Adding fractional dose constraints...")
        self.fractional_dose_constraint()
        logger.model("Adding constraint 3b...")
        self.initialize_constraint_3b()
        logger.model("Adding constraint 3c1...")
        self.initialize_constraint_3c1()
        logger.model("Adding constraint 3c2...")
        self.initialize_constraint_3c2()
        logger.model("Adding constraint 3d...")
        self.initialize_constraint_3d()
        logger.model("Adding constraint 3e...")
        self.initialize_constraint_3e()
        logger.model("Adding constraint 3f...")
        self.initialize_constraint_3f()
        logger.model("Setting objective function...")
        self.initialize_objective()
        logger.model("Model building completed.")
    
    def initialize_objective(self):
        """
        Initializes the objective.
        """
        self._model.setObjective(self._d_underbar + self._optimization_parameters.lam * self._d_underbar_F, GRB.MAXIMIZE)
    
    def solve(self) -> None:
        """
        Solves the model.
        """
        logger.model(f"Solving model...")
        self._model.optimize()
        self._model_status = self._model.Status
        if self._model_status == GRB.OPTIMAL:
            logger.model("Optimal solution found.")
            self._model.write("my_model.sol")  # Saves the solution (variable values)
            self._model.write("my_model.lp")   # Saves the model in readable LP format

        elif self._model_status == GRB.INFEASIBLE:
            logger.model("Model is infeasible. Computing IIS...")
            self._model.computeIIS()
            self._model.write("my_model.ilp")
            self._model.write("my_model.lp")

        else:
            logger.model(f"Solver ended with status code: {self._model_status}")
    
    def get_solution(self) -> dict[str, np.ndarray] | None:
        """
        Returns the solution.
        """
        if self._model_status == GRB.OPTIMAL:
            return {
                "beamlet_intensities": self._x.X,
                "d_underbar_F": self._d_underbar_F.X,
                "d_underbar": self._d_underbar.X,
            }