import logging

def configure_logging():
    # Define custom TEST level
    TEST_LEVEL = 25
    PREPROCESS_LEVEL = 26
    MODEL_LEVEL = 27
    SOLVER_LEVEL = 28
    GUROBI_LEVEL = 29
    logging.addLevelName(TEST_LEVEL, "TEST")
    logging.addLevelName(PREPROCESS_LEVEL, "PREPROCESS")
    logging.addLevelName(MODEL_LEVEL, "MODEL")
    logging.addLevelName(SOLVER_LEVEL, "SOLVER")
    logging.addLevelName(GUROBI_LEVEL, "GUROBI")

    def preprocess(self, message, *args, **kwargs):
        if self.isEnabledFor(PREPROCESS_LEVEL):
            self._log(PREPROCESS_LEVEL, message, args, **kwargs)

    def model(self, message, *args, **kwargs):
        if self.isEnabledFor(MODEL_LEVEL):
            self._log(MODEL_LEVEL, message, args, **kwargs)

    def solver(self, message, *args, **kwargs):
        if self.isEnabledFor(SOLVER_LEVEL):
            self._log(SOLVER_LEVEL, message, args, **kwargs)

    def test(self, message, *args, **kwargs):
        if self.isEnabledFor(TEST_LEVEL):
            self._log(TEST_LEVEL, message, args, **kwargs)
    
    def gurobi(self, message, *args, **kwargs):
        if self.isEnabledFor(GUROBI_LEVEL):
            self._log(GUROBI_LEVEL, message, args, **kwargs)

    logging.Logger.test = test
    logging.Logger.preprocess = preprocess
    logging.Logger.model = model
    logging.Logger.solver = solver
    logging.Logger.gurobi = gurobi
    # ANSI color codes
    CYAN = "\033[96m"
    PURPLE = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            levelname = record.levelname
            if levelname == "PREPROCESS":
                record.levelname = f"{CYAN}{levelname}{RESET}"
            elif levelname == "MODEL":
                record.levelname = f"{PURPLE}{levelname}{RESET}"
            elif levelname == "SOLVER":
                record.levelname = f"{YELLOW}{levelname}{RESET}"
            elif levelname == "TEST":
                record.levelname = f"{GREEN}{levelname}{RESET}"
            elif levelname == "GUROBI":
                record.levelname = f"{RED}{levelname}{RESET}"
            elif levelname == "INFO":
                record.levelname = f"{RESET}{levelname}{RESET}"
            return super().format(record)

    formatter = CustomFormatter("[%(levelname)s] %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
