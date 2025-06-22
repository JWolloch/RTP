from preprocessor import Preprocessor
from model import Model
from config import OptimizationParameters, GammaParameters, ProjectionParameters
from utils import MemoryMonitor, save_run_results
from logger_config import configure_logging
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging()
    
    # Initialize preprocessor
    preprocessor = Preprocessor("data/liverEx_2.mat")
    preprocessor.check_phi_bounds()
    preprocessor.print_min_max_projections()
    preprocessor.print_sample_projections()
    
    # Create optimization parameters
    optimization_params = OptimizationParameters()
    
    # Create and build the model
    model = Model(preprocessor, optimization_params, debug=optimization_params.debug)

    # Start monitoring
    memory_monitor = MemoryMonitor(interval=0.1)
    memory_monitor.start()

    # --- Start model solving section ---
    if optimization_params.row_generation:
        if optimization_params.debug:
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
            solve_time_seconds=model._solver_time,
            peak_memory_mb=peak_memory_mb,
            solution_dict=solution
        )
    else:
        logger.solver("No optimal solution found. Check the model status.")