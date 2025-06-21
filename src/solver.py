from preprocessor import Preprocessor
from model import Model
from config import OptimizationParameters
from logger_config import configure_logging
import numpy as np

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
        model.build_model_without_homogeneity_constraints()
        found_feasible_solution, total_constraints_added, objective_value_per_iteration, c1_constraints_added_per_iteration, c2_constraints_added_per_iteration = model.row_generation_model_solver()
    else:
        model.build_full_model()
        model.solve_full_model()

    # Get and display results
    solution = model.get_solution()
    if solution:
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Minimum fractional dose (d_underbar_F): {solution['d_underbar_F']:.6f}")
        print(f"Minimum total dose (d_underbar): {solution['d_underbar']:.6f}")
        if optimization_params.row_generation:
            if found_feasible_solution:
                print("Found feasible solution.")
            else:
                print("No feasible solution found.")
            print(f"Total constraints added: {total_constraints_added}")
            print(f"Objective value per iteration: {objective_value_per_iteration}")
            print(f"Constraints added per iteration: {c1_constraints_added_per_iteration}")
            print(f"Constraints added per iteration: {c2_constraints_added_per_iteration}")
    else:
        print("No optimal solution found. Check the model status.")