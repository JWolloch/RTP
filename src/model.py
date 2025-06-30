from preprocessor import Preprocessor
from config import OptimizationParameters
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csc_matrix, hstack, eye, diags
import os
import logging
import heapq

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
        self._H_3 = self._preprocessor.H_3_voxels.shape[0] # H_3 is the number of voxels in organ 3
        self._N = self._optimization_parameters.N # N is the number of fractions (this implementation supports only N=2)
        self._mu_F = self._optimization_parameters.mu_F # mu_F - fractional homogeneity parameter
        self._d_bar_organ_1 = self._optimization_parameters.d_bar_organ_1 # d_bar_organ_1 is the maximum radiation dose for organ 1
        self._d_bar_organ_2 = self._optimization_parameters.d_bar_organ_2 # d_bar_organ_2 is the maximum radiation dose for organ 2
        self._d_bar_organ_3 = self._optimization_parameters.d_bar_organ_3 # d_bar_organ_3 is the maximum radiation dose for organ 3
        self._d_bar_F_organ_1 = self._optimization_parameters.d_bar_F_organ_1 # d_bar_F_organ_1 is the maximum fractional radiation dose for organ 1
        self._d_bar_F_organ_2 = self._optimization_parameters.d_bar_F_organ_2 # d_bar_F_organ_2 is the maximum fractional radiation dose for organ 2
        self._d_bar_F_organ_3 = self._optimization_parameters.d_bar_F_organ_3 # d_bar_F_organ_3 is the maximum fractional radiation dose for organ 3
        self._eps = self._optimization_parameters.eps # eps is the tolerance for the fractional dose constraints

        self._max_constraint_addition = self._optimization_parameters.max_constraint_addition
        self._env = gp.Env(empty=True)
        self._env.setParam(GRB.Param.OutputFlag, 1)
        self._env.start()

        # Create model using this environment
        self._model = gp.Model(env=self._env)
        self._folder_name = f"{self._optimization_parameters.solution_method.name}_{self._optimization_parameters.n_most_violated_constraints}_{self._optimization_parameters.max_constraint_addition}"
        os.makedirs(f"final_results/mu_F_{self._optimization_parameters.mu_F}/{self._folder_name}", exist_ok=True)

        # Write Gurobi logs to file (but NOT to console)
        self._model.setParam(GRB.Param.LogFile, f"final_results/mu_F_{self._optimization_parameters.mu_F}/{self._folder_name}/gurobi.log")
        self._model.setParam(GRB.Param.DualReductions, 0)
        self._model.setParam(GRB.Param.Method, self._optimization_parameters.solution_method.value)


        self._x = self.initialize_beamlet_intensity_variables()
        self._d_underbar_F = self.initialize_minimum_fractional_dose_variable()
        self._d_underbar = self.initialize_minimum_total_dose_variable()

        self._dose_tumor_voxels, self._dose_healthy_voxels_organ_1, self._dose_healthy_voxels_organ_2, self._dose_healthy_voxels_organ_3 = self.initialize_fractional_dose_variables()

        self._model_status = None
        self._solver_time = None
        
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
        dose_tumor_voxels = self._model.addMVar(shape=(self._N, self._T), lb=0.0, name="fractional_dose_tumor_voxels")
        dose_healthy_voxels_organ_1 = self._model.addMVar(shape=(self._N, self._H_1), lb=0.0, name="fractional_dose_healthy_voxels_organ_1")
        dose_healthy_voxels_organ_2 = self._model.addMVar(shape=(self._N, self._H_2), lb=0.0, name="fractional_dose_healthy_voxels_organ_2")
        dose_healthy_voxels_organ_3 = self._model.addMVar(shape=(self._N, self._H_3), lb=0.0, name="fractional_dose_healthy_voxels_organ_3")
        logger.model(f"Initialized {self._N}x{self._T} fractional dose auxiliary variables for tumor voxels and {self._N}x{self._H_1} fractional dose auxiliary variables for healthy voxels in organ 1 and {self._N}x{self._H_2} fractional dose auxiliary variables for healthy voxels in organ 2 and {self._N}x{self._H_3} fractional dose auxiliary variables for healthy voxels in organ 3")
        return dose_tumor_voxels, dose_healthy_voxels_organ_1, dose_healthy_voxels_organ_2, dose_healthy_voxels_organ_3
    
    def fractional_dose_constraint(self) -> None:
        """
        Initializes fractional dose constraints.
        """
        I_tumor = -1 * eye(self._T)
        D_tumor_sparse = csc_matrix(self._D[:self._T])
        blocks = [I_tumor, D_tumor_sparse]

        A_tumor = hstack(blocks, format="csc")

        tumor_var_list_1 = self._dose_tumor_voxels[0].tolist() + self._x[0].tolist()
        y_tumor_1 = gp.MVar.fromlist(tumor_var_list_1)
        tumor_var_list_2 = self._dose_tumor_voxels[1].tolist() + self._x[1].tolist()
        y_tumor_2 = gp.MVar.fromlist(tumor_var_list_2)

        self._model.addMConstr(A_tumor, y_tumor_1, GRB.EQUAL, np.zeros(self._T), name="fractional_dose_constraint_tumor_1")
        self._model.addMConstr(A_tumor, y_tumor_2, GRB.EQUAL, np.zeros(self._T), name="fractional_dose_constraint_tumor_2")
        
        #Now we do the same for healthy voxels in organ 1
        I_healthy_organ_1 = -1 * eye(self._H_1)
        D_healthy_organ_1_sparse = csc_matrix(self._D[self._T:self._T + self._H_1])
        blocks = [I_healthy_organ_1, D_healthy_organ_1_sparse]

        A_healthy_organ_1 = hstack(blocks, format="csc")

        healthy_organ_1_var_list_1 = self._dose_healthy_voxels_organ_1[0].tolist() + self._x[0].tolist()
        y_healthy_organ_1_1 = gp.MVar.fromlist(healthy_organ_1_var_list_1)
        healthy_organ_1_var_list_2 = self._dose_healthy_voxels_organ_1[1].tolist() + self._x[1].tolist()
        y_healthy_organ_1_2 = gp.MVar.fromlist(healthy_organ_1_var_list_2)
        
        self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_1, GRB.EQUAL, np.zeros(self._H_1), name="fractional_dose_constraint_healthy_organ_1_1")
        self._model.addMConstr(A_healthy_organ_1, y_healthy_organ_1_2, GRB.EQUAL, np.zeros(self._H_1), name="fractional_dose_constraint_healthy_organ_1_2")

        #Now we do the same for healthy voxels in organ 2
        I_healthy_organ_2 = -1 * eye(self._H_2)
        D_healthy_organ_2_sparse = csc_matrix(self._D[self._T + self._H_1:self._T + self._H_1 + self._H_2])
        blocks = [I_healthy_organ_2, D_healthy_organ_2_sparse]

        A_healthy_organ_2 = hstack(blocks, format="csc")

        healthy_organ_2_var_list_1 = self._dose_healthy_voxels_organ_2[0].tolist() + self._x[0].tolist()
        y_healthy_organ_2_1 = gp.MVar.fromlist(healthy_organ_2_var_list_1)
        healthy_organ_2_var_list_2 = self._dose_healthy_voxels_organ_2[1].tolist() + self._x[1].tolist()
        y_healthy_organ_2_2 = gp.MVar.fromlist(healthy_organ_2_var_list_2)

        self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_1, GRB.EQUAL, np.zeros(self._H_2), name="fractional_dose_constraint_healthy_organ_2_1")
        self._model.addMConstr(A_healthy_organ_2, y_healthy_organ_2_2, GRB.EQUAL, np.zeros(self._H_2), name="fractional_dose_constraint_healthy_organ_2_2")
    
        #Now we do the same for healthy voxels in organ 3
        I_healthy_organ_3 = -1 * eye(self._H_3)
        D_healthy_organ_3_sparse = csc_matrix(self._D[self._T + self._H_1 + self._H_2:self._T + self._H_1 + self._H_2 + self._H_3])
        blocks = [I_healthy_organ_3, D_healthy_organ_3_sparse]

        A_healthy_organ_3 = hstack(blocks, format="csc")

        healthy_organ_3_var_list_1 = self._dose_healthy_voxels_organ_3[0].tolist() + self._x[0].tolist()
        y_healthy_organ_3_1 = gp.MVar.fromlist(healthy_organ_3_var_list_1)
        healthy_organ_3_var_list_2 = self._dose_healthy_voxels_organ_3[1].tolist() + self._x[1].tolist()
        y_healthy_organ_3_2 = gp.MVar.fromlist(healthy_organ_3_var_list_2)

        self._model.addMConstr(A_healthy_organ_3, y_healthy_organ_3_1, GRB.EQUAL, np.zeros(self._H_3), name="fractional_dose_constraint_healthy_organ_3_1")
        self._model.addMConstr(A_healthy_organ_3, y_healthy_organ_3_2, GRB.EQUAL, np.zeros(self._H_3), name="fractional_dose_constraint_healthy_organ_3_2")

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


        self._model.addMConstr(A, y_1, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3b_1")
        self._model.addMConstr(A, y_2, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3b_2")
    
    def initialize_constraint_3c1(self):
        """
        Initializes the constraint 3c1.
        """
        logger.model(f"Building constraint 3c1 for {self._T} tumor voxels...")
        
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
        logger.model(f"Building constraint 3c2 for {self._T} tumor voxels...")

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
        A1 = -1 * csc_matrix(np.ones((self._T, 1)))
        A2 = diags(self._preprocessor.phi_underbar_1)
        A3 = diags(self._preprocessor.phi_underbar_2)
        blocks = [A1, A2, A3]
        A = hstack(blocks, format="csc")

        var_list = [self._d_underbar] + self._dose_tumor_voxels[0].tolist() + self._dose_tumor_voxels[1].tolist()
        y = gp.MVar.fromlist(var_list)

        self._model.addMConstr(A, y, GRB.GREATER_EQUAL, np.zeros(self._T), name="constraint_3d")
    
    def initialize_constraint_3e(self):
        """
        Initializes the constraint 3e.
        """
        A_organ_1 = eye(self._H_1)
        A_organ_2 = eye(self._H_2)
        A_organ_3 = eye(self._H_3)

        #======== Organ 1 =========
        y_1 = self._dose_healthy_voxels_organ_1[0]
        y_2 = self._dose_healthy_voxels_organ_1[1]

        self._model.addMConstr(A_organ_1, y_1, GRB.LESS_EQUAL, self._d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3e_1")
        self._model.addMConstr(A_organ_1, y_2, GRB.LESS_EQUAL, self._d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3e_2")

        #======== Organ 2 =========
        z_1 = self._dose_healthy_voxels_organ_2[0]
        z_2 = self._dose_healthy_voxels_organ_2[1]

        self._model.addMConstr(A_organ_2, z_1, GRB.LESS_EQUAL, self._d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3e_3")
        self._model.addMConstr(A_organ_2, z_2, GRB.LESS_EQUAL, self._d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3e_4")
        
        #======== Organ 3 =========
        w_1 = self._dose_healthy_voxels_organ_3[0]
        w_2 = self._dose_healthy_voxels_organ_3[1]

        self._model.addMConstr(A_organ_3, w_1, GRB.LESS_EQUAL, self._d_bar_F_organ_3 * np.ones(self._H_3), name="constraint_3e_5")
        self._model.addMConstr(A_organ_3, w_2, GRB.LESS_EQUAL, self._d_bar_F_organ_3 * np.ones(self._H_3), name="constraint_3e_6")
    
    def initialize_constraint_3f(self):
        """
        Initializes the constraint 3f.
        """
        I_organ_1 = eye(self._H_1)
        I_organ_2 = eye(self._H_2)
        I_organ_3 = eye(self._H_3)

        A_organ_1 = hstack([I_organ_1, I_organ_1], format="csc")
        A_organ_2 = hstack([I_organ_2, I_organ_2], format="csc")
        A_organ_3 = hstack([I_organ_3, I_organ_3], format="csc")
        #======== Organ 1 =========
        y = self._dose_healthy_voxels_organ_1[0].tolist() + self._dose_healthy_voxels_organ_1[1].tolist()

        self._model.addMConstr(A_organ_1, y, GRB.LESS_EQUAL, self._d_bar_F_organ_1 * np.ones(self._H_1), name="constraint_3f_1")

        #======== Organ 2 =========
        z = self._dose_healthy_voxels_organ_2[0].tolist() + self._dose_healthy_voxels_organ_2[1].tolist()

        self._model.addMConstr(A_organ_2, z, GRB.LESS_EQUAL, self._d_bar_F_organ_2 * np.ones(self._H_2), name="constraint_3f_2")

        #======== Organ 3 =========
        w = self._dose_healthy_voxels_organ_3[0].tolist() + self._dose_healthy_voxels_organ_3[1].tolist()

        self._model.addMConstr(A_organ_3, w, GRB.LESS_EQUAL, self._d_bar_F_organ_3 * np.ones(self._H_3), name="constraint_3f_3")

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
        constraints_to_add_1 = [] #list of tuples (violation, v, u)
        constraints_to_add_2 = [] #list of tuples (violation, v, u)
        logger.model(f"Building constraint 3c1 for {self._T} tumor voxels...")
        
        for v in range(self._T):
            if v % 100 == 0:  # Log progress every 100 voxels
                logger.model(f"Constraint 3c1 progress: {v}/{self._T} voxels processed")

            v1_old, v2_old = self._voxels_already_considered_c1[f"{v}"]
            #======== Fraction 1 =========

            indices_to_consider_1 = np.setdiff1d(self._indices, v1_old) #avoiding re-evaluating already added constraints
            k1 = indices_to_consider_1.shape[0]

            if k1 == 0:
                continue #skip if no indices to consider

            A1 = self._preprocessor.phi_bar_1[v] * csc_matrix(np.ones((k1, 1)))
            A2 = -self._mu_F * diags(self._preprocessor.M_3c1_1[indices_to_consider_1, v].toarray().flatten())

            blocks = [A1, A2]
            A = hstack(blocks, format="csc")

            indices = np.concatenate((np.array([v]), indices_to_consider_1))
            y_1 = self._dose_tumor_voxels[0][indices]
            y_1_value = y_1.X

            constraint_lhs_1 = (A @ y_1_value).flatten() #constraint lhs

            mask_1 = constraint_lhs_1 > self._eps #mask of positive constraint lhs
            violated_indices_1 = np.where(mask_1)[0]
            violated_lhs_1 = constraint_lhs_1[violated_indices_1]

            l_1 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_1.shape[0])

            if l_1 == 0:
                continue #skip if no violated constraints

            top_l_indices_1 = np.argsort(violated_lhs_1)[-l_1:]
            most_violated_indices_1 = violated_indices_1[top_l_indices_1]
            most_violated_values_1 = violated_lhs_1[top_l_indices_1]

            constraints_to_add_1.extend(
            [(float(violation), v, indices_to_consider_1[u_idx]) for violation, u_idx in zip(most_violated_values_1, most_violated_indices_1)]
            )
            #======== Fraction 2 =========
            indices_to_consider_2 = np.setdiff1d(self._indices, v2_old) #avoiding re-evaluating already added constraints
            k2 = indices_to_consider_2.shape[0]

            if k2 == 0:
                continue #skip if no indices to consider

            B1 = self._preprocessor.phi_bar_2[v] * csc_matrix(np.ones((k2, 1)))
            B2 = -self._mu_F * diags(self._preprocessor.M_3c1_2[indices_to_consider_2, v].toarray().flatten())

            blocks = [B1, B2]
            B = hstack(blocks, format="csc")

            indices = np.concatenate((np.array([v]), indices_to_consider_2))
            y_2 = self._dose_tumor_voxels[1][indices]
            y_2_value = y_2.X

            constraint_lhs_2 = (B @ y_2_value).flatten() #constraint lhs

            mask_2 = constraint_lhs_2 > self._eps #mask of positive constraint lhs
            violated_indices_2 = np.where(mask_2)[0]
            violated_lhs_2 = constraint_lhs_2[violated_indices_2]

            l_2 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_2.shape[0])
            if l_2 == 0:
                continue #skip if no violated constraints

            top_l_indices_2 = np.argsort(violated_lhs_2)[-l_2:]
            most_violated_indices_2 = violated_indices_2[top_l_indices_2]
            most_violated_values_2 = violated_lhs_2[top_l_indices_2]

            constraints_to_add_2.extend(
            [(float(violation), v, indices_to_consider_2[u_idx]) for violation, u_idx in zip(most_violated_values_2, most_violated_indices_2)]
            )
        
        most_violated_constraints_1 = heapq.nlargest(self._optimization_parameters.max_constraint_addition, constraints_to_add_1, key=lambda x: x[0])

        v_voxel_indices_1 = np.array([constraint[1] for constraint in most_violated_constraints_1], dtype=int) #casting to int to avoid type error for empty array
        u_voxel_indices_1 = np.array([constraint[2] for constraint in most_violated_constraints_1], dtype=int)

        if v_voxel_indices_1.shape[0] > 0:
        
            left_diagonal_matrix_1 = diags(self._preprocessor.phi_bar_1[v_voxel_indices_1])
            vals = np.asarray(self._preprocessor.M_3c1_1[u_voxel_indices_1, v_voxel_indices_1]).flatten()
            right_diagonal_matrix_1 = -self._mu_F * diags(vals)

            A_1 = hstack([left_diagonal_matrix_1, right_diagonal_matrix_1], format="csc")

            indices_1 = np.concatenate((v_voxel_indices_1, u_voxel_indices_1))
            z_1 = self._dose_tumor_voxels[0][indices_1]

            self._model.addMConstr(A_1, z_1, GRB.LESS_EQUAL, np.zeros(A_1.shape[0]), name="constraint_3c1_1")

        most_violated_constraints_2 = heapq.nlargest(self._optimization_parameters.max_constraint_addition, constraints_to_add_2, key=lambda x: x[0])

        v_voxel_indices_2 = np.array([constraint[1] for constraint in most_violated_constraints_2], dtype=int) #casting to int to avoid type error for empty array
        u_voxel_indices_2 = np.array([constraint[2] for constraint in most_violated_constraints_2], dtype=int)

        if v_voxel_indices_2.shape[0] > 0:

            left_diagonal_matrix_2 = diags(self._preprocessor.phi_bar_2[v_voxel_indices_2])
            vals = np.asarray(self._preprocessor.M_3c1_2[u_voxel_indices_2, v_voxel_indices_2]).flatten()
            right_diagonal_matrix_2 = -self._mu_F * diags(vals)
            
            A_2 = hstack([left_diagonal_matrix_2, right_diagonal_matrix_2], format="csc")
            
            indices_2 = np.concatenate((v_voxel_indices_2, u_voxel_indices_2))
            z_2 = self._dose_tumor_voxels[1][indices_2]
            
            self._model.addMConstr(A_2, z_2, GRB.LESS_EQUAL, np.zeros(A_2.shape[0]), name="constraint_3c1_2")

        for constraint in most_violated_constraints_1:
            old_already_considered_voxels_1, old_already_considered_voxels_2 = self._voxels_already_considered_c1[f"{constraint[1]}"]
            new_already_considered_voxels_1 = np.append(old_already_considered_voxels_1, constraint[2])
            self._voxels_already_considered_c1[f"{constraint[1]}"] = (new_already_considered_voxels_1, old_already_considered_voxels_2)
        
        for constraint in most_violated_constraints_2:
            old_already_considered_voxels_1, old_already_considered_voxels_2 = self._voxels_already_considered_c1[f"{constraint[1]}"]
            new_already_considered_voxels_2 = np.append(old_already_considered_voxels_2, constraint[2])
            self._voxels_already_considered_c1[f"{constraint[1]}"] = (old_already_considered_voxels_1, new_already_considered_voxels_2)
        
        number_of_constraints_added = len(most_violated_constraints_1) + len(most_violated_constraints_2)

        logger.model("Constraint 3c1 completed.")

        return number_of_constraints_added

    def evaluate_constraint_3c2(self) -> int:
        """
        Initializes the constraint 3c2.
        """
        constraints_to_add_1 = [] #list of tuples (violation, v, u)
        constraints_to_add_2 = [] #list of tuples (violation, v, u)
        logger.model(f"Building constraint 3c2 for {self._T} tumor voxels...")

        for v in range(self._T):
            if v % 100 == 0:  # Log progress every 100 voxels
                logger.model(f"Constraint 3c2 progress: {v}/{self._T} voxels processed")

            v1_old, v2_old = self._voxels_already_considered_c2[f"{v}"]
            #======== Fraction 1 =========
            indices_to_consider_1 = np.setdiff1d(self._indices, v1_old) #avoiding re-evaluating already added constraints

            if indices_to_consider_1.shape[0] == 0:
                continue #skip if no indices to consider

            A1 = self._preprocessor.M_3c2_1[indices_to_consider_1, v]
            A2 = -self._mu_F * diags(self._preprocessor.phi_underbar_1[indices_to_consider_1])

            blocks = [A1, A2]
            A = hstack(blocks, format="csc")

            indices = np.concatenate((np.array([v]), indices_to_consider_1))
            y_1 = self._dose_tumor_voxels[0][indices]
            y_1_value = y_1.X

            constraint_lhs_1 = (A @ y_1_value).flatten() #constraint lhs

            mask_1 = constraint_lhs_1 > self._eps #mask of positive constraint lhs
            violated_indices_1 = np.where(mask_1)[0]
            violated_lhs_1 = constraint_lhs_1[violated_indices_1]

            l_1 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_1.shape[0])

            if l_1 == 0:
                continue #skip if no violated constraints

            top_l_indices_1 = np.argsort(violated_lhs_1)[-l_1:]
            most_violated_indices_1 = violated_indices_1[top_l_indices_1]
            most_violated_values_1 = violated_lhs_1[top_l_indices_1]

            constraints_to_add_1.extend(
            [(float(violation), v, indices_to_consider_1[u_idx]) for violation, u_idx in zip(most_violated_values_1, most_violated_indices_1)]
            )

            #======== Fraction 2 =========
            indices_to_consider_2 = np.setdiff1d(self._indices, v2_old) #avoiding re-evaluating already added constraints

            if indices_to_consider_2.shape[0] == 0:
                continue #skip if no indices to consider

            B1 = self._preprocessor.M_3c2_2[indices_to_consider_2, v]
            B2 = -self._mu_F * diags(self._preprocessor.phi_underbar_2[indices_to_consider_2])

            blocks = [B1, B2]
            B = hstack(blocks, format="csc")

            indices = np.concatenate((np.array([v]), indices_to_consider_2))
            y_2 = self._dose_tumor_voxels[1][indices]
            y_2_value = y_2.X

            constraint_lhs_2 = (B @ y_2_value).flatten() #constraint lhs

            mask_2 = constraint_lhs_2 > self._eps #mask of positive constraint lhs
            violated_indices_2 = np.where(mask_2)[0]
            violated_lhs_2 = constraint_lhs_2[violated_indices_2]

            l_2 = min(self._optimization_parameters.n_most_violated_constraints, violated_lhs_2.shape[0])

            if l_2 == 0:
                continue #skip if no violated constraints

            top_l_indices_2 = np.argsort(violated_lhs_2)[-l_2:]
            most_violated_indices_2 = violated_indices_2[top_l_indices_2]
            most_violated_values_2 = violated_lhs_2[top_l_indices_2]

            constraints_to_add_2.extend(
            [(float(violation), v, indices_to_consider_2[u_idx]) for violation, u_idx in zip(most_violated_values_2, most_violated_indices_2)]
            )

        most_violated_constraints_1 = heapq.nlargest(self._optimization_parameters.max_constraint_addition, constraints_to_add_1, key=lambda x: x[0])

        v_voxel_indices_1 = np.array([constraint[1] for constraint in most_violated_constraints_1], dtype=int) #casting to int to avoid type error for empty array
        u_voxel_indices_1 = np.array([constraint[2] for constraint in most_violated_constraints_1], dtype=int)

        if v_voxel_indices_1.shape[0] > 0:
        
            left_vals = np.asarray(self._preprocessor.M_3c2_1[u_voxel_indices_1, v_voxel_indices_1]).flatten()
            left_diagonal_matrix_1 = diags(left_vals)

            right_vals = self._preprocessor.phi_underbar_1[u_voxel_indices_1]
            right_diagonal_matrix_1 = -self._mu_F * diags(right_vals)

            A_1 = hstack([left_diagonal_matrix_1, right_diagonal_matrix_1], format="csc")

            indices_1 = np.concatenate((v_voxel_indices_1, u_voxel_indices_1))
            z_1 = self._dose_tumor_voxels[0][indices_1]

            self._model.addMConstr(A_1, z_1, GRB.LESS_EQUAL, np.zeros(A_1.shape[0]), name="constraint_3c2_1")

        
        most_violated_constraints_2 = heapq.nlargest(self._optimization_parameters.max_constraint_addition, constraints_to_add_2, key=lambda x: x[0])

        v_voxel_indices_2 = np.array([constraint[1] for constraint in most_violated_constraints_2], dtype=int) #casting to int to avoid type error for empty array
        u_voxel_indices_2 = np.array([constraint[2] for constraint in most_violated_constraints_2], dtype=int)

        if v_voxel_indices_2.shape[0] > 0:

            left_vals = np.asarray(self._preprocessor.M_3c2_2[u_voxel_indices_2, v_voxel_indices_2]).flatten()
            left_diagonal_matrix_2 = diags(left_vals)

            right_vals = self._preprocessor.phi_underbar_2[u_voxel_indices_2]
            right_diagonal_matrix_2 = -self._mu_F * diags(right_vals)

            A_2 = hstack([left_diagonal_matrix_2, right_diagonal_matrix_2], format="csc")

            indices_2 = np.concatenate((v_voxel_indices_2, u_voxel_indices_2))
            z_2 = self._dose_tumor_voxels[1][indices_2]

            self._model.addMConstr(A_2, z_2, GRB.LESS_EQUAL, np.zeros(A_2.shape[0]), name="constraint_3c2_2")

        for constraint in most_violated_constraints_1:
            old_already_considered_voxels_1, old_already_considered_voxels_2 = self._voxels_already_considered_c2[f"{constraint[1]}"]
            new_already_considered_voxels_1 = np.append(old_already_considered_voxels_1, constraint[2])
            self._voxels_already_considered_c2[f"{constraint[1]}"] = (new_already_considered_voxels_1, old_already_considered_voxels_2)

        for constraint in most_violated_constraints_2:
            old_already_considered_voxels_1, old_already_considered_voxels_2 = self._voxels_already_considered_c2[f"{constraint[1]}"]
            new_already_considered_voxels_2 = np.append(old_already_considered_voxels_2, constraint[2])
            self._voxels_already_considered_c2[f"{constraint[1]}"] = (old_already_considered_voxels_1, new_already_considered_voxels_2)
        
        number_of_constraints_added = len(most_violated_constraints_1) + len(most_violated_constraints_2)
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
            self._model.update()
            logger.model(f"--- Row Generation Iteration {iteration + 1} ---")
            logger.model(f"mu_F: {self._optimization_parameters.mu_F}")
            logger.model(f"max_constraint_addition: {self._optimization_parameters.max_constraint_addition}")
            logger.model(f"n_most_violated_constraints: {self._optimization_parameters.n_most_violated_constraints}")
            if iteration == 0:
                logger.model("Initial solve - no homogeneity constraints added.")
            logger.model(f"Invoking Gurobi solver - solution method: {self._optimization_parameters.solution_method.name}...")
            logger.model(f"Model has {self._model.NumVars} variables and {self._model.NumConstrs} constraints.")
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
                logger.model(f"Number of voxels already considered: {len(self._voxels_already_considered_c1)}")
                logger.model(f"Number of voxels already considered: {len(self._voxels_already_considered_c2)}")
                found_feasible_solution = True
                logger.model("No more violated constraints found. Terminating row generation.")
                break

            iteration += 1

        # Final solve to finalize solution
        logger.model("Final solve with all constraints...")
        self._model.optimize()
        self._model_status = self._model.Status

        output_dir = f"final_solutions/mu_F_{self._optimization_parameters.mu_F}_max_constraint_addition_{self._optimization_parameters.max_constraint_addition}_n_most_violated_constraints_{self._optimization_parameters.n_most_violated_constraints}"
        os.makedirs(output_dir, exist_ok=True)  

        if self._model_status == GRB.OPTIMAL:
            logger.model("Row generation: Optimal solution found.")
            self._model.write(f"{output_dir}/rowgen_model.sol")
        elif self._model_status == GRB.INFEASIBLE:
            logger.model("Row generation model is infeasible. Computing IIS...")
            self._model.computeIIS()
            self._model.write(f"{output_dir}/rowgen_model.ilp")
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
                "organ3_voxels_bio-adjusted_dosages_fraction_1": self._dose_healthy_voxels_organ_3[0].X,
                "tumor_voxels_bio-adjusted_dosages_fraction_2": self._dose_tumor_voxels[1].X,
                "organ1_voxels_bio-adjusted_dosages_fraction_2": self._dose_healthy_voxels_organ_1[1].X,
                "organ2_voxels_bio-adjusted_dosages_fraction_2": self._dose_healthy_voxels_organ_2[1].X,
                "organ3_voxels_bio-adjusted_dosages_fraction_2": self._dose_healthy_voxels_organ_3[1].X,
                "d_underbar_F": self._d_underbar_F.X,
                "d_underbar": self._d_underbar.X,
            }