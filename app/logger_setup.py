"""
Logging configuration.
Sets up structured logging for the Churn Prediction API.
"""
import logging
import sys
def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging and return a named logger.
      1: Set up basic logging with level INFO using logging.basicConfig()
      2: Create a named logger using logging.getLogger() and return it
    Returns:
        logging.Logger: Configured logger for the application.
    """
    #   1: Set up basic logging with level INFO
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    #   2: Create and return a named logger
    logger = logging.getLogger("churn_prediction_api")
    logger.info("Logging initialised – level=INFO")
    return logger