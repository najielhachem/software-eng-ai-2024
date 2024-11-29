# src/ml_data_pipeline/main.py
import argparse

from loguru import logger

from ml_data_pipeline.config import load_config
from ml_data_pipeline.core import load_pipeline
from ml_data_pipeline.data_loader import DataLoaderFactory

# Configure loguru to log to a file and console
logger.add("logs/pipeline.log", rotation="500 MB")  # Log rotation at 500 MB
parser = argparse.ArgumentParser(
    description="Run the ML data pipeline with specified configuration."
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration YAML file."
)


def main() -> None:
    logger.info("Parsing command line arguments.")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}.")

    logger.info("Loading configuration.")
    config = load_config(args.config)
    logger.info("Loaded configuration successfully.")
    logger.debug(f"Configuration: {config}")

    # Use DataLoaderFactory to load data
    try:
        data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
        data = data_loader.load_data(config.data_loader.file_path)
        logger.info("Data loaded successfully.")
        logger.debug(f"Data: {data.head()}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    inference_pipeline = load_pipeline(config.transformation, config.model)
    inference_pipeline.run(data)


if __name__ == "__main__":
    main()
