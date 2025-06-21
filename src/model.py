from preprocessor import Preprocessor
from config import OptimizationParameters
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csc_matrix, hstack, eye, diags
import os
import logging


logger = logging.getLogger(__name__)

class Model:
    def __init__(self, preprocessor: Preprocessor, optimization_parameters: OptimizationParameters, debug: bool=False):
        self._preprocessor = preprocessor
        self._optimization_parameters = optimization_parameters
        self._debug = debug
        self._debug_n = self._optimization_parameters.debug_n
        self._D = self._preprocessor.D # D is a matrix of shape (number_of_voxels, number_of_beamlets), number_of_voxels includes healthy organ as well as tumor voxels
        self._m = self._D.shape[0] # m is the total number of voxels
        self._n = self._D.shape[1] # n is the number of beamlets
        self._T = self._preprocessor.phi_hat.shape[0] # T is the number of tumor voxels
        self._H_1 = self._preprocessor.H_1_voxels.shape[0] # H_1 is the number of voxels in organ 1
        self._H_2 = self._preprocessor.H_2_voxels.shape[0] # H_2 is the number of voxels in organ 2
        self._N = self._optimization_parameters.N # N is the number of fractions (this implementation supports only N=2)
        self._mu_F = self._optimization_parameters.mu_F # mu_F - fractional homogeneity parameter
        self._d_bar_F = self._optimization_parameters.d_bar_F # d_bar_F is the maximum fractional radiation dose

        self._env = gp.Env(empty=True)
        self._env.setParam(GRB.Param.OutputFlag, 1)
        self._env.start()

        # Create model using this environment
        self._model = gp.Model(env=self._env)
        self._folder_name = f"{self._optimization_parameters.solution_method.name}_{self._optimization_parameters.n_most_violated_constraints}"
        os.makedirs(f"results/{self._folder_name}", exist_ok=True)

        # Write Gurobi logs to file (but NOT to console)
        self._model.setParam(GRB.Param.LogFile, f"results/{self._folder_name}/gurobi.log")
        self._model.setParam(GRB.Param.DualReductions, 0)


        self._x = self.initialize_beamlet_intensity_variables()
        self._d_underbar_F = self.initialize_minimum_fractional_dose_variable()
        self._d_underbar = self.initialize_minimum_total_dose_variable()

        self._dose_tumor_voxels, self._dose_healthy_voxels_organ_1, self._dose_healthy_voxels_organ_2 = self.initialize_fractional_dose_variables()

        self._model_status = None
        self._solver_time = None
        
        if self._debug:
            self._indices = np.arange(self._debug_n, dtype=int)
            self._voxels_already_considered_c1 = {f"{v}": (np.array([], dtype=int), np.array([], dtype=int)) for v in range(self._debug_n)}
            self._voxels_already_considered_c2 = {f"{v}": (np.array([], dtype=int), np.array([], dtype=int)) for v in range(self._debug_n)}
        else:
            self._indices = np.arange(self._T, dtype=int)
            self._voxels_already_considered_c1 = {f"{v}": (np.array([], dtype=int), np.array([], dtype=int)) for v in range(self._T)}
            self._voxels_already_considered_c2 = {f"{v}": (np.array([], dtype=int), np.array([], dtype=int)) for v in range(self._T)}

    def initialize_beamlet_intensity_variables(self):
        x = self._model.addMVar(shape=(self._N, self._n), lb=0.0, name="x")
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
            dose_tumor_voxels = self._model.addMVar(shape=(self._N, self._debug_n), lb=0.0, name="fractional_dose_tumor_voxels")
            dose_healthy_voxels_organ_1 = self._model.addMVar(shape=(self._N, self._debug_n), lb=0.0, name="fractional_dose_healthy_voxels_organ_1")
            dose_healthy_voxels_organ_2 = self._model.addMVar(shape=(self._N, self._debug_n), lb=0.0, name="fractional_dose_healthy_voxels_organ_2")
            logger.model(f"Initialized {self._N}x{self._debug_n} fractional dose auxiliary variables for tumor voxels and {self._N}x{self._debug_n} fractional dose auxiliary variables for healthy voxels in organ 1 and {self._N}x{self._debug_n} fractional dose auxiliary variables for healthy voxels in organ 2")
        else:
            dose_tumor_voxels = self._model.addMVar(shape=(self._N, self._T), lb=0.0, name="fractional_dose_tumor_voxels")
            dose_healthy_voxels_organ_1 = self._model.addMVar(shape=(self._N, self._H_1), lb=0.0, name="fractional_dose_healthy_voxels_organ_1")
            dose_healthy_voxels_organ_2 = self._model.addMVar(shape=(self._N, self._H_2), lb=0.0, name="fractional_dose_healthy_voxels_organ_2")
        logger.model(f"Initialized {self._N}x{self._T} fractional dose auxiliary variables for tumor voxels and {self._N}x{self._H_1} fractional dose auxiliary variables for healthy voxels in organ 1 and {self._N}x{self._H_2} fractional dose auxiliary variables for healthy voxels in organ 2")
        return dose_tumor_voxels, dose_healthy_voxels_organ_1, dose_healthy_voxels_organ_2
    
    def fractional_dose_constraint(self) -> None:
        """
        Initializes fractional dose constraints.
        """
        #We start with tumor voxels
        if self._debug:
            I_tumor = -1 * eye(self._debug_n)
            D_tumor_sparse = csc_matrix(self._D[:self._debug_n])
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
            self._model.addMConstr(A_tumor, y_tumor_1, GRB.EQUAL, np.zeros(self._debug_n), name="fractional_dose_constraint_tumor_1")
            self._model.addMConstr(A_tumor, y_tumor_2, GRB.EQUAL, np.zeros(self._debug_n), name="fractional_dose_constraint_tumor_2")
        else:
            self._model.addMConstr(A_tumor, y_tumor_1, GRB.EQUAL, np.zeros(self._T), name="fractional_dose_constraint_tumor_1")
            self._model.addMConstr(A_tumor, y_tumor_2, GRB.EQUAL, np.zeros(self._T), name="fractional_dose_constraint_tumor_2")
        
        #Now we do the same for healthy voxels in organ 1
        if self._debug:
            I_healthy_organ_1 = -1 * eye(self._debug_n)
            D_healthy_organ_1_sparse = csc_matrix(self._D[self._T:self._T + self._debug_n])
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
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_1, GRB.EQUAL, np.zeros(self._debug_n), name="fractional_dose_constraint_healthy_organ_1_1")
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_2, GRB.EQUAL, np.zeros(self._debug_n), name="fractional_dose_constraint_healthy_organ_1_2")
        else:
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_1, GRB.EQUAL, np.zeros(self._H_1), name="fractional_dose_constraint_healthy_organ_1_1")
            self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_2, GRB.EQUAL, np.zeros(self._H_1), name="fractional_dose_constraint_healthy_organ_1_2")

        #Now we do the same for healthy voxels in organ 2
        if self._debug:
            I_healthy_organ_2 = -1 * eye(self._debug_n)
            D_healthy_organ_2_sparse = csc_matrix(self._D[self._T + self._H_1:self._T + self._H_1 + self._debug_n])
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
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_1, GRB.EQUAL, np.zeros(self._debug_n), name="fractional_dose_constraint_healthy_organ_2_1")
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_2, GRB.EQUAL, np.zeros(self._debug_n), name="fractional_dose_constraint_healthy_organ_2_2")
        else:
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_1, GRB.EQUAL, np.zeros(self._H_2), name="fractional_dose_constraint_healthy_organ_2_1")
            self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_2, GRB.EQUAL, np.zeros(self._H_2), name="fractional_dose_constraint_healthy_organ_2_2")
    
    def initialize_constraint_3b(self):
        """
        Initializes the constraint 3b.
        """
        if self._debug:
            A1 = -1 * csc_matrix(np.ones((self._debug_n, 1)))
            A2 = csc_matrix(np.diag(self._preprocessor.phi_underbar_1[:self._debug_n]))
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
            self._model.addMConstr(A, y_1, GRB.GREATER_EQUAL, np.zeros(self._debug_n), name="constraint_3b_1")
            self._model.addMConstr(A, y_2, GRB.GREATER_EQUAL, np.zeros(self._debug_n), name="constraint_3b_2")
        else:
            self._model.addMConstr(A, y_1, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3b_1")
            self._model.addMConstr(A, y_2, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3b_2")
    
    def initialize_constraint_3c1(self):
        """
        Initializes the constraint 3c1.
        """
        if self._debug:
            logger.model(f"Building constraint 3c1 for {self._debug_n} tumor voxels...")
        else:
            logger.model(f"Building constraint 3c1 for {self._T} tumor voxels...")
        
        if self._debug:
            for v in range(self._debug_n):
                logger.model(f"Constraint 3c1 progress: {v}/{self._debug_n} voxels processed")
                indices = np.concatenate((np.array([v]), self._indices))
                #======== Fraction 1 =========
                A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((self._debug_n, 1)))
                A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[:, v][:self._debug_n].toarray().flatten())

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                y_1 = self._dose_tumor_voxels[0][indices]

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(self._debug_n), name=f"constraint_3c1_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((self._debug_n, 1)))
                B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[:, v][:self._debug_n].toarray().flatten())

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                y_2 = self._dose_tumor_voxels[1][indices]

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(self._debug_n), name=f"constraint_3c1_2_{v}")
            logger.model("Constraint 3c1 completed.")

        else:

            for v in range(self._T):
                if v % 100 == 0:  # Log progress every 100 voxels
                    logger.model(f"Constraint 3c1 progress: {v}/{self._T} voxels processed")
                indices = np.concatenate((np.array([v]), self._indices))
                #======== Fraction 1 =========
                A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((self._T, 1)))
                A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[:, v].toarray().flatten())

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                y_1 = self._dose_tumor_voxels[0][indices]

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c1_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((self._T, 1)))
                B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[:, v].toarray().flatten())

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                y_2 = self._dose_tumor_voxels[1][indices]

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c1_2_{v}")
            logger.model("Constraint 3c1 completed.")
    
    def initialize_constraint_3c2(self):
        """
        Initializes the constraint 3c2.
        """
        if self._debug:
            logger.model(f"Building constraint 3c2 for {self._debug_n} tumor voxels...")
        else:
            logger.model(f"Building constraint 3c2 for {self._T} tumor voxels...")

        if self._debug:
            for v in range(self._debug_n):
                logger.model(f"Constraint 3c2 progress: {v}/{self._debug_n} voxels processed")
                indices = np.concatenate((np.array([v]), self._indices))
                #======== Fraction 1 =========
                A1 = self._preprocessor.M_3c2_1[:, v][:self._debug_n]
                A2 = -self._mu_F * diags(self._preprocessor.phi_bar_1[:self._debug_n])

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                y_1 = self._dose_tumor_voxels[0][indices]

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(self._debug_n), name=f"constraint_3c2_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.M_3c2_2[:, v][:self._debug_n]
                B2 = -self._mu_F * diags(self._preprocessor.phi_bar_2[:self._debug_n])

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                y_2 = self._dose_tumor_voxels[1][indices]

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(self._debug_n), name=f"constraint_3c2_2_{v}")
            logger.model("Constraint 3c2 completed.")
        else:
            for v in range(self._T):
                if v % 100 == 0:  # Log progress every 100 voxels
                    logger.model(f"Constraint 3c2 progress: {v}/{self._T} voxels processed")
                indices = np.concatenate((np.array([v]), self._indices))
                #======== Fraction 1 =========
                A1 = self._preprocessor.M_3c2_1[:, v]
                A2 = -self._mu_F * diags(self._preprocessor.phi_bar_1)

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                y_1 = self._dose_tumor_voxels[0][indices]

                self._model.addMConstr(A, y_1, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c2_1_{v}")

                #======== Fraction 2 =========
                B1 = self._preprocessor.M_3c2_2[:, v]
                B2 = -self._mu_F * diags(self._preprocessor.phi_bar_2)

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                y_2 = self._dose_tumor_voxels[1][indices]

                self._model.addMConstr(B, y_2, GRB.LESS_EQUAL, np.zeros(self._T), name=f"constraint_3c2_2_{v}")
            logger.model("Constraint 3c2 completed.")
        
    def initialize_constraint_3d(self):
        """
        Initializes the constraint 3d.
        """
        if self._debug:
            A1 = -1 * csc_matrix(np.ones((self._debug_n, 1)))
            A2 = diags(self._preprocessor.phi_underbar_1[:self._debug_n])
            A3 = diags(self._preprocessor.phi_underbar_2[:self._debug_n])
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
            self._model.addMConstr(A, y, GRB.GREATER_EQUAL, np.zeros(self._debug_n), name="constraint_3d")
        else:
            self._model.addMConstr(A, y, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3d")
    
    def initialize_constraint_3e(self):
        """
        Initializes the constraint 3e.
        """
        if self._debug:
            A_organ_1 = eye(self._debug_n)
            A_organ_2 = eye(self._debug_n)
        else:
            A_organ_1 = eye(self._H_1)
            A_organ_2 = eye(self._H_2)

        #======== Organ 1 =========
        y_1 = self._dose_healthy_voxels_organ_1[0]
        y_2 = self._dose_healthy_voxels_organ_1[1]

        if self._debug:
            self._model.addMConstr(A_organ_1, y_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(self._debug_n), name="constraint_3e_1")
            self._model.addMConstr(A_organ_1, y_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(self._debug_n), name="constraint_3e_2")
        else:
            self._model.addMConstr(A_organ_1, y_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3e_1")
            self._model.addMConstr(A_organ_1, y_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3e_2")

        #======== Organ 2 =========
        z_1 = self._dose_healthy_voxels_organ_2[0]
        z_2 = self._dose_healthy_voxels_organ_2[1]

        if self._debug:
            self._model.addMConstr(A_organ_2, z_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(self._debug_n), name="constraint_3e_3")
            self._model.addMConstr(A_organ_2, z_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(self._debug_n), name="constraint_3e_4")
        else:
            self._model.addMConstr(A_organ_2, z_1, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3e_3")
            self._model.addMConstr(A_organ_2, z_2, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3e_4")
    
    def initialize_constraint_3f(self):
        """
        Initializes the constraint 3f.
        """
        if self._debug:
            I_organ_1 = eye(self._debug_n)
            I_organ_2 = eye(self._debug_n)
        else:
            I_organ_1 = eye(self._H_1)
            I_organ_2 = eye(self._H_2)

        A_organ_1 = hstack([I_organ_1, I_organ_1], format="csc")
        A_organ_2 = hstack([I_organ_2, I_organ_2], format="csc")

        #======== Organ 1 =========
        y = self._dose_healthy_voxels_organ_1[0].tolist() + self._dose_healthy_voxels_organ_1[1].tolist()

        if self._debug:
            self._model.addMConstr(A_organ_1, y, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_1 * np.ones(self._debug_n), name="constraint_3f_1")
        else:
            self._model.addMConstr(A_organ_1, y, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_1 * np.ones(self._H_1), name="constraint_3f_1")

        #======== Organ 2 =========
        z = self._dose_healthy_voxels_organ_2[0].tolist() + self._dose_healthy_voxels_organ_2[1].tolist()

        if self._debug:
            self._model.addMConstr(A_organ_2, z, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_2 * np.ones(self._debug_n), name="constraint_3f_2")
        else:
            self._model.addMConstr(A_organ_2, z, GRB.LESS_EQUAL, self._optimization_parameters.d_bar_organ_2 * np.ones(self._H_2), name="constraint_3f_2")
    
    def initialize_objective(self):
        """
        Initializes the objective.
        """
        self._model.setObjective(self._d_underbar + self._optimization_parameters.lam * self._d_underbar_F, GRB.MAXIMIZE)

    def build_full_model(self) -> None:
        """
        Builds the model in full.
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
    
    def solve_full_model(self) -> None:
        """
        Solves the full model.
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
    
    def build_model_without_homogeneity_constraints(self) -> None:
        """
        Builds the model without homogeneity constraints.
        """
        logger.model("Building model constraints...")
        logger.model("Adding fractional dose constraints...")
        self.fractional_dose_constraint()
        logger.model("Adding constraint 3b...")
        self.initialize_constraint_3b()
        logger.model("Adding constraint 3d...")
        self.initialize_constraint_3d()
        logger.model("Adding constraint 3e...")
        self.initialize_constraint_3e()
        logger.model("Adding constraint 3f...")
        self.initialize_constraint_3f()
        logger.model("Setting objective function...")
        self.initialize_objective()
        logger.model("Model building completed.")
    
    def evaluate_constraint_3c1(self) -> int:
        """
        Given the current solution, evaluates it and adds the most violated constraints to the model.
        """
        number_of_constraints_added = 0
        if self._debug:
            logger.model(f"Building constraint 3c1 for {self._debug_n} tumor voxels...")
        else:
            logger.model(f"Building constraint 3c1 for {self._T} tumor voxels...")
        
        if self._debug:
            for v in range(self._debug_n):
                logger.model(f"Constraint 3c1 progress: {v}/{self._debug_n} voxels processed")
                v1_old, v2_old = self._voxels_already_considered_c1[f"{v}"]
                #======== Fraction 1 =========
                indices_to_consider_1 = np.setdiff1d(self._indices, v1_old) #avoiding re-evaluating already added constraints
                k1 = indices_to_consider_1.shape[0]
                A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((k1, 1)))
                A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[indices_to_consider_1, v].toarray().flatten())


                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_1))
                y_1 = self._dose_tumor_voxels[0][indices]
                y_1_value = y_1.X

                constraint_lhs_1 = (A @ y_1_value).flatten() #constraint lhs

                mask_1 = constraint_lhs_1 > 0 #mask of positive constraint lhs
                violated_indices_1 = np.where(mask_1)[0]
                violated_lhs_1 = constraint_lhs_1[violated_indices_1]

                l_1 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_1.shape[0])
                top_l_indices_1 = np.argsort(violated_lhs_1)[-l_1:]
                most_violated_indices_1 = violated_indices_1[top_l_indices_1]

                v1_new = np.concatenate((v1_old, indices_to_consider_1))

                r_1 = most_violated_indices_1.shape[0]

                if r_1 > 0:
                    number_of_constraints_added += r_1
                    self._model.addMConstr(A[most_violated_indices_1], y_1, GRB.LESS_EQUAL, np.zeros(r_1), name=f"constraint_3c1_1_{v}")


                #======== Fraction 2 =========
                indices_to_consider_2 = np.setdiff1d(self._indices, v2_old) #avoiding re-evaluating already added constraints
                k2 = indices_to_consider_2.shape[0]
                B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((k2, 1)))
                B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[indices_to_consider_2, v].toarray().flatten())

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_2))
                y_2 = self._dose_tumor_voxels[1][indices]
                y_2_value = y_2.X

                constraint_lhs_2 = (B @ y_2_value).flatten() #constraint lhs

                mask_2 = constraint_lhs_2 > 0 #mask of positive constraint lhs
                violated_indices_2 = np.where(mask_2)[0]
                violated_lhs_2 = constraint_lhs_2[violated_indices_2]

                l_2 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_2.shape[0])
                top_l_indices_2 = np.argsort(violated_lhs_2)[-l_2:]
                most_violated_indices_2 = violated_indices_2[top_l_indices_2]

                v2_new = np.concatenate((v2_old, indices_to_consider_2))

                r_2 = most_violated_indices_2.shape[0]

                if r_2 > 0:
                    number_of_constraints_added += r_2
                    self._model.addMConstr(B[most_violated_indices_2], y_2, GRB.LESS_EQUAL, np.zeros(r_2), name=f"constraint_3c1_2_{v}")
            
                self._voxels_already_considered_c1[f"{v}"] = (v1_new, v2_new)

            logger.model("Constraint 3c1 completed.")

        else:

            for v in range(self._T):
                if v % 100 == 0:  # Log progress every 100 voxels
                    logger.model(f"Constraint 3c1 progress: {v}/{self._T} voxels processed")

                v1_old, v2_old = self._voxels_already_considered_c1[f"{v}"]
                #======== Fraction 1 =========
                indices_to_consider_1 = np.setdiff1d(self._indices, v1_old) #avoiding re-evaluating already added constraints
                k1 = indices_to_consider_1.shape[0]
                A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((k1, 1)))
                A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[indices_to_consider_1, v].toarray().flatten())

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_1))
                y_1 = self._dose_tumor_voxels[0][indices]
                y_1_value = y_1.X

                constraint_lhs_1 = (A @ y_1_value).flatten() #constraint lhs

                mask_1 = constraint_lhs_1 > 0 #mask of positive constraint lhs
                violated_indices_1 = np.where(mask_1)[0]
                violated_lhs_1 = constraint_lhs_1[violated_indices_1]

                l_1 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_1.shape[0])
                top_l_indices_1 = np.argsort(violated_lhs_1)[-l_1:]
                most_violated_indices_1 = violated_indices_1[top_l_indices_1]

                v1_new = np.concatenate((v1_old, indices_to_consider_1))

                r_1 = most_violated_indices_1.shape[0]

                if r_1 > 0:
                    number_of_constraints_added += r_1
                    self._model.addMConstr(A[most_violated_indices_1], y_1, GRB.LESS_EQUAL, np.zeros(r_1), name=f"constraint_3c1_1_{v}")

                #======== Fraction 2 =========
                indices_to_consider_2 = np.setdiff1d(self._indices, v2_old) #avoiding re-evaluating already added constraints
                k2 = indices_to_consider_2.shape[0]
                B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((k2, 1)))
                B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[indices_to_consider_2, v].toarray().flatten())

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_2))
                y_2 = self._dose_tumor_voxels[1][indices]
                y_2_value = y_2.X

                constraint_lhs_2 = (B @ y_2_value).flatten() #constraint lhs

                mask_2 = constraint_lhs_2 > 0 #mask of positive constraint lhs
                violated_indices_2 = np.where(mask_2)[0]
                violated_lhs_2 = constraint_lhs_2[violated_indices_2]

                l_2 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_2.shape[0])
                top_l_indices_2 = np.argsort(violated_lhs_2)[-l_2:]
                most_violated_indices_2 = violated_indices_2[top_l_indices_2]

                v2_new = np.concatenate((v2_old, indices_to_consider_2))

                r_2 = most_violated_indices_2.shape[0]

                if r_2 > 0:
                    number_of_constraints_added += r_2
                    self._model.addMConstr(B[most_violated_indices_2], y_2, GRB.LESS_EQUAL, np.zeros(r_2), name=f"constraint_3c1_2_{v}")

                self._voxels_already_considered_c1[f"{v}"] = (v1_new, v2_new)

            logger.model("Constraint 3c1 completed.")

        return number_of_constraints_added

    def evaluate_constraint_3c2(self) -> int:
        """
        Initializes the constraint 3c2.
        """
        number_of_constraints_added = 0
        if self._debug:
            logger.model(f"Building constraint 3c2 for {self._debug_n} tumor voxels...")
        else:
            logger.model(f"Building constraint 3c2 for {self._T} tumor voxels...")

        if self._debug:
            for v in range(self._debug_n):
                logger.model(f"Constraint 3c2 progress: {v}/{self._debug_n} voxels processed")
                v1_old, v2_old = self._voxels_already_considered_c2[f"{v}"]

                #======== Fraction 1 =========
                indices_to_consider_1 = np.setdiff1d(self._indices, v1_old) #avoiding re-evaluating already added constraints
                A1 = self._preprocessor.M_3c2_1[indices_to_consider_1, v]
                A2 = -self._mu_F * diags(self._preprocessor.phi_bar_1[indices_to_consider_1])

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_1))
                y_1 = self._dose_tumor_voxels[0][indices]
                y_1_value = y_1.X

                constraint_lhs_1 = (A @ y_1_value).flatten() #constraint lhs

                mask_1 = constraint_lhs_1 > 0 #mask of positive constraint lhs
                violated_indices_1 = np.where(mask_1)[0]
                violated_lhs_1 = constraint_lhs_1[violated_indices_1]

                l_1 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_1.shape[0])
                top_l_indices_1 = np.argsort(violated_lhs_1)[-l_1:]
                most_violated_indices_1 = violated_indices_1[top_l_indices_1]

                v1_new = np.concatenate((v1_old, indices_to_consider_1))

                r_1 = most_violated_indices_1.shape[0]

                if r_1 > 0:
                    number_of_constraints_added += r_1
                    self._model.addMConstr(A[most_violated_indices_1], y_1, GRB.LESS_EQUAL, np.zeros(r_1), name=f"constraint_3c2_1_{v}")

                #======== Fraction 2 =========
                indices_to_consider_2 = np.setdiff1d(self._indices, v2_old) #avoiding re-evaluating already added constraints
                B1 = self._preprocessor.M_3c2_2[indices_to_consider_2, v]
                B2 = -self._mu_F * diags(self._preprocessor.phi_bar_2[indices_to_consider_2])

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_2))
                y_2 = self._dose_tumor_voxels[1][indices]
                y_2_value = y_2.X

                constraint_lhs_2 = (B @ y_2_value).flatten() #constraint lhs

                mask_2 = constraint_lhs_2 > 0 #mask of positive constraint lhs
                violated_indices_2 = np.where(mask_2)[0]
                violated_lhs_2 = constraint_lhs_2[violated_indices_2]

                l_2 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_2.shape[0])
                top_l_indices_2 = np.argsort(violated_lhs_2)[-l_2:]
                most_violated_indices_2 = violated_indices_2[top_l_indices_2]

                v2_new = np.concatenate((v2_old, indices_to_consider_2))

                r_2 = most_violated_indices_2.shape[0]

                if r_2 > 0:
                    number_of_constraints_added += r_2
                    self._model.addMConstr(B[most_violated_indices_2], y_2, GRB.LESS_EQUAL, np.zeros(r_2), name=f"constraint_3c2_2_{v}")

                self._voxels_already_considered_c2[f"{v}"] = (v1_new, v2_new)

            logger.model("Constraint 3c2 completed.")
        else:
            for v in range(self._T):
                if v % 100 == 0:  # Log progress every 100 voxels
                    logger.model(f"Constraint 3c2 progress: {v}/{self._T} voxels processed")

                v1_old, v2_old = self._voxels_already_considered_c2[f"{v}"]
                #======== Fraction 1 =========
                indices_to_consider_1 = np.setdiff1d(self._indices, v1_old) #avoiding re-evaluating already added constraints
                A1 = self._preprocessor.M_3c2_1[indices_to_consider_1, v]
                A2 = -self._mu_F * diags(self._preprocessor.phi_bar_1[indices_to_consider_1])

                blocks = [A1, A2]
                A = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_1))
                y_1 = self._dose_tumor_voxels[0][indices]
                y_1_value = y_1.X

                constraint_lhs_1 = (A @ y_1_value).flatten() #constraint lhs

                mask_1 = constraint_lhs_1 > 0 #mask of positive constraint lhs
                violated_indices_1 = np.where(mask_1)[0]
                violated_lhs_1 = constraint_lhs_1[violated_indices_1]

                l_1 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_1.shape[0])
                top_l_indices_1 = np.argsort(violated_lhs_1)[-l_1:]
                most_violated_indices_1 = violated_indices_1[top_l_indices_1]

                v1_new = np.concatenate((v1_old, indices_to_consider_1))

                r_1 = most_violated_indices_1.shape[0]

                if r_1 > 0:
                    number_of_constraints_added += r_1
                    self._model.addMConstr(A[most_violated_indices_1], y_1, GRB.LESS_EQUAL, np.zeros(r_1), name=f"constraint_3c2_1_{v}")

                #======== Fraction 2 =========
                indices_to_consider_2 = np.setdiff1d(self._indices, v2_old) #avoiding re-evaluating already added constraints
                B1 = self._preprocessor.M_3c2_2[indices_to_consider_2, v]
                B2 = -self._mu_F * diags(self._preprocessor.phi_bar_2[indices_to_consider_2])

                blocks = [B1, B2]
                B = hstack(blocks, format="csc")

                indices = np.concatenate((np.array([v]), indices_to_consider_2))
                y_2 = self._dose_tumor_voxels[1][indices]
                y_2_value = y_2.X

                constraint_lhs_2 = (B @ y_2_value).flatten() #constraint lhs

                mask_2 = constraint_lhs_2 > 0 #mask of positive constraint lhs
                violated_indices_2 = np.where(mask_2)[0]
                violated_lhs_2 = constraint_lhs_2[violated_indices_2]

                l_2 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_2.shape[0])
                top_l_indices_2 = np.argsort(violated_lhs_2)[-l_2:]
                most_violated_indices_2 = violated_indices_2[top_l_indices_2]

                v2_new = np.concatenate((v2_old, indices_to_consider_2))

                r_2 = most_violated_indices_2.shape[0]

                if r_2 > 0:
                    number_of_constraints_added += r_2
                    self._model.addMConstr(B[most_violated_indices_2], y_2, GRB.LESS_EQUAL, np.zeros(r_2), name=f"constraint_3c2_2_{v}")

                self._voxels_already_considered_c2[f"{v}"] = (v1_new, v2_new)

            logger.model("Constraint 3c2 completed.")

        return number_of_constraints_added
    
    def row_generation_model_solver(self) -> None:
        """
        Solves the model using a row generation approach.
        Adds the most violated constraints iteratively until all are satisfied.
        """
        logger.model("Starting row generation solver...")

        max_iterations = self._optimization_parameters.max_row_generation_iterations
        iteration = 0
        objective_value_per_iteration = []
        c1_constraints_added_per_iteration = []
        c2_constraints_added_per_iteration = []
        total_constraints_added = 0
        found_feasible_solution = False

        start_time = time.time()

        while iteration < max_iterations:
            logger.model(f"--- Row Generation Iteration {iteration + 1} ---")
            logger.model(f"Invoking Gurobi solver - solution method: {self._optimization_parameters.solution_method.name}...")
            logger.model(f"At most {self._optimization_parameters.n_most_violated_constraints} violated constraints will be added per voxel, per fraction, per iteration.")
            self._model.optimize()
            self._model_status = self._model.Status

            if self._model_status == GRB.OPTIMAL:
                logger.model(f"Iteration {iteration + 1} completed: Optimal solution found. Evaluating for violations...")
            elif self._model_status == GRB.INFEASIBLE:
                logger.model("Model became infeasible. Aborting.")
                break
            else:
                logger.model(f"Solver status code: {self._model_status}. Aborting.")
                break

            # Evaluate and add violated constraints
            c1_constraints_added_per_iteration.append(self.evaluate_constraint_3c1())  # Internally updates count
            c2_constraints_added_per_iteration.append(self.evaluate_constraint_3c2())  # Internally updates count

            added_this_iter = c1_constraints_added_per_iteration[-1] + c2_constraints_added_per_iteration[-1]
            total_constraints_added += added_this_iter
            
            if self._model_status == GRB.OPTIMAL:
                objective_value_per_iteration.append(self._model.ObjVal)
            else:
                objective_value_per_iteration.append(None)

            if self._model_status == GRB.OPTIMAL:
                logger.model(f"Iteration {iteration + 1}: {added_this_iter} constraints added. Objective value: {self._model.ObjVal}")
            else:
                logger.warning(f"Iteration {iteration + 1}: {added_this_iter} constraints added. Objective value not available.")


            # Stop if no new constraints were added
            if added_this_iter == 0:
                found_feasible_solution = True
                logger.model("No more violated constraints found. Terminating row generation.")
                break

            iteration += 1

        # Final solve to finalize solution
        logger.model("Final solve with all constraints...")
        self._model.optimize()
        self._model_status = self._model.Status

        if self._model_status == GRB.OPTIMAL:
            logger.model("Row generation: Optimal solution found.")
            if self._debug:
                self._model.write(f"results/{self._folder_name}/debug_rowgen_model.sol")
            else:
                self._model.write(f"results/{self._folder_name}/rowgen_model.sol")
        elif self._model_status == GRB.INFEASIBLE:
            logger.model("Row generation model is infeasible. Computing IIS...")
            self._model.computeIIS()
            if self._debug:
                self._model.write(f"results/{self._folder_name}/debug_rowgen_model.ilp")
            else:
                self._model.write(f"results/{self._folder_name}/rowgen_model.ilp")
        else:
            logger.model(f"Solver ended with status code: {self._model_status}")

        self._solver_time = time.time() - start_time

        return found_feasible_solution, total_constraints_added, objective_value_per_iteration, c1_constraints_added_per_iteration, c2_constraints_added_per_iteration


    
    def get_solution(self) -> dict[str, np.ndarray] | None:
        """
        Returns the solution.
        """
        if self._model_status == GRB.OPTIMAL:
            return {
                "beamlet_intensities": self._x.X,
                "tumor_voxels_bio-adjusted_dosages_fraction_1": self._dose_tumor_voxels[0].X,
                "organ1_voxels_bio-adjusted_dosages_fraction_1": self._dose_healthy_voxels_organ_1[0].X,
                "organ2_voxels_bio-adjusted_dosages_fraction_1": self._dose_healthy_voxels_organ_2[0].X,
                "tumor_voxels_bio-adjusted_dosages_fraction_2": self._dose_tumor_voxels[1].X,
                "organ1_voxels_bio-adjusted_dosages_fraction_2": self._dose_healthy_voxels_organ_1[1].X,
                "organ2_voxels_bio-adjusted_dosages_fraction_2": self._dose_healthy_voxels_organ_2[1].X,
                "d_underbar_F": self._d_underbar_F.X,
                "d_underbar": self._d_underbar.X,
            }