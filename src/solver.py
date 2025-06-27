from preprocessor import Preprocessor
from model import Model
from config import OptimizationParameters, GammaParameters, ProjectionParameters
from utils import MemoryMonitor, save_run_results
from logger_config import configure_logging
import logging
from itertools import product
from typing import Any

logger = logging.getLogger(__name__)

def run_solver(param_dict: dict[str, Any], preprocessor: Preprocessor):
    # Create optimization parameters
    optimization_params = OptimizationParameters.from_dict(param_dict)
    
    # Create and build the model
    model = Model(preprocessor, optimization_params)

    # Start monitoring
    memory_monitor = MemoryMonitor(interval=0.1)
    memory_monitor.start()

    # --- Start model solving section ---
    if optimization_params.row_generation:
        if preprocessor.debug:
            logger.solver("In debug mode")
            
        logger.solver("Solving model in row generation mode")

        model.build_model_without_homogeneity_constraints()

        found_feasible_solution, total_constraints_added, objective_value_per_iteration, c1_constraints_added_per_iteration, c2_constraints_added_per_iteration = model.row_generation_model_solver()
    else:
        model.build_full_model()
        model.solve_full_model()
    # --- End model solving section ---

    # Stop monitoring
    memory_monitor.stop()
    peak_memory_mb = memory_monitor.peak_memory / (1024**2)

    logger.solver("Process Completed")
    logger.solver(f"Solver time: {model._solver_time:.2f} seconds")
    logger.solver(f"Peak memory usage during model solution: {peak_memory_mb:.2f} MB")

    # Get and display results
    solution = model.get_solution()
    if solution:
        logger.solver("\n=== OPTIMIZATION RESULTS ===")
        logger.solver(f"Minimum fractional dose (d_underbar_F): {solution['d_underbar_F']:.6f}")
        logger.solver(f"Minimum total dose (d_underbar): {solution['d_underbar']:.6f}")
        if optimization_params.row_generation:
            if found_feasible_solution:
                logger.solver("Found feasible solution.")
            else:
                logger.solver("No feasible solution found.")
            logger.solver(f"Total constraints added: {total_constraints_added}")
            logger.solver(f"Objective value per iteration: {objective_value_per_iteration}")
            logger.solver(f"Constraints added per iteration 3C1: {c1_constraints_added_per_iteration}")
            logger.solver(f"Constraints added per iteration 3C2: {c2_constraints_added_per_iteration}")
        # Save results
        save_run_results(
            gamma_params=GammaParameters(),
            proj_params=ProjectionParameters(),
            opt_params=optimization_params,
            found_feasible_solution=found_feasible_solution,
            objective_value_per_iteration=objective_value_per_iteration,
            total_constraints_added=total_constraints_added,
            c1_constraints_added_per_iteration=c1_constraints_added_per_iteration,
            c2_constraints_added_per_iteration=c2_constraints_added_per_iteration,
            solve_time_seconds=model._solver_time,
            peak_memory_mb=peak_memory_mb,
            solution_dict=solution
        )
    else:
        logger.solver("No optimal solution found. Check the model status.")


if __name__ == "__main__":
    configure_logging()
    
    # Initialize preprocessor
    preprocessor = Preprocessor("data/liverEx_2.mat", debug=True, debug_n=1000)
    preprocessor.check_phi_bounds()
    preprocessor.print_min_max_projections()
    preprocessor.print_sample_projections()

    mu_F_vals = [1.2, 1.25, 1.3, 1.35, 1.4]
    max_constraint_addition_vals = [10**3, 10**4, 10**5, 10**10]
    n_most_violated_constraints_vals = [2, 5, 10]

    param_combinations = list(product(mu_F_vals, max_constraint_addition_vals, n_most_violated_constraints_vals))
    total_runs = len(param_combinations)

    for i, (mu_F, max_constraint_addition, n_most_violated_constraints) in enumerate(param_combinations, 1):
        logger.solver(f"\n>>> Starting run {i}/{total_runs} with mu_F={mu_F}, max_add={max_constraint_addition}, top_violated={n_most_violated_constraints}")
        
        param_dict = {
            "mu_F": mu_F,
            "max_constraint_addition": max_constraint_addition,
            "n_most_violated_constraints": n_most_violated_constraints
        }
        try:
            run_solver(param_dict, preprocessor)
        except Exception as e:
            logger.solver(f"Run failed for {param_dict}: {e}")
            continue