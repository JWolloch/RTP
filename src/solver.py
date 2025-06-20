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
        model.build_full_model()
        model.solve_full_model()
    else:
        model.build_model_without_homogeneity_constraints()
        found_feasible_solution, total_constraints_added, objective_value_per_iteration, c1_constraints_added_per_iteration, c2_constraints_added_per_iteration = model.row_generation_model_solver()

        if found_feasible_solution:
            model.build_model_with_homogeneity_constraints()
            model.solve()
        else:
            print("No feasible solution found. Check the model status.")

    
    # Get and display results
    solution = model.get_solution()
    if solution:
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Minimum fractional dose (d_underbar_F): {solution['d_underbar_F']:.6f}")
        print(f"Minimum total dose (d_underbar): {solution['d_underbar']:.6f}")
        print(f"Beamlet intensities shape: {solution['beamlet_intensities'].shape}")
        print(f"Beamlet intensities - Fraction 1 min/max: {solution['beamlet_intensities'][0].min():.6f}/{solution['beamlet_intensities'][0].max():.6f}")
        print(f"Beamlet intensities - Fraction 2 min/max: {solution['beamlet_intensities'][1].min():.6f}/{solution['beamlet_intensities'][1].max():.6f}")
        
        # Calculate total dose for tumor voxels
        total_tumor_dose = np.sum(solution['beamlet_intensities'], axis=0)
        print(f"Total tumor dose - min/max: {total_tumor_dose.min():.6f}/{total_tumor_dose.max():.6f}")
    else:
        print("No optimal solution found. Check the model status.") 