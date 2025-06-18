import logging

def configure_logging():
    # Define custom TEST level
    TEST_LEVEL = 25
    logging.addLevelName(TEST_LEVEL, "TEST")

    def test(self, message, *args, **kwargs):
        if self.isEnabledFor(TEST_LEVEL):
            self._log(TEST_LEVEL, message, args, **kwargs)

    logging.Logger.test = test

    # ANSI color codes
    CYAN = "\033[96m"
    PURPLE = "\033[95m"
    RESET = "\033[0m"

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            levelname = record.levelname
            if levelname == "INFO":
                record.levelname = f"{CYAN}{levelname}{RESET}"
            elif levelname == "TEST":
                record.levelname = f"{PURPLE}{levelname}{RESET}"
            return super().format(record)

    formatter = CustomFormatter("[%(levelname)s] %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
