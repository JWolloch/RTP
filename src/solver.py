from preprocessor import Preprocessor
from logger_config import configure_logging

if __name__ == "__main__":
    configure_logging()
    preprocessor = Preprocessor("data/liverEx_2.mat")
    preprocessor.check_phi_bounds()
    preprocessor.print_min_max_projections() 
    preprocessor.print_sample_projections() 