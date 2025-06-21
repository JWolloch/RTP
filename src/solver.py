from preprocessor import Preprocessor
from model import Model
from config import OptimizationParameters
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

    if optimization_params.row_generation:
        if optimization_params.debug:
            logger.solver("In debug mode")
            
        logger.solver("Solving model in row generation mode")

        model.build_model_without_homogeneity_constraints()

        found_feasible_solution, total_constraints_added, objective_value_per_iteration, c1_constraints_added_per_iteration, c2_constraints_added_per_iteration = model.row_generation_model_solver()
    else:
        model.build_full_model()
        model.solve_full_model()

    logger.solver("Process Completed")
    logger.solver(f"Solver time: {model._solver_time:.2f} seconds")
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
    else:
        logger.solver("No optimal solution found. Check the model status.")