import argparse

import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml_data_pipeline.config import load_config
from ml_data_pipeline.data_loader import DataLoaderFactory
from ml_data_pipeline.data_transformer import TransformerFactory
from ml_data_pipeline.models.factory import ModelFactory

logger.add("logs/training.log", rotation="500 MB")
parser = argparse.ArgumentParser(
    description="Run the ML data pipeline training with specified configuration."
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

    # Initialize MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    mlflow.autolog()

    with mlflow.start_run():
        try:
            # Log parameters
            mlflow.log_param("model_type", config.model.type)
            mlflow.log_param("model_parameters", config.model.params)

            # Load and transform data
            data_loader = DataLoaderFactory.get_data_loader(
                config.data_loader.file_type
            )
            data = data_loader.load_data(config.data_loader.file_path)
            transformer = TransformerFactory.get_transformer(
                config.transformation.scaling_method
            )
            transformed_data = transformer.transform(data)

            # Prepare train-test split
            X = transformed_data.drop(columns=["target"])
            y = transformed_data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train the model
            model = ModelFactory.get_model(config.model)
            model.train(X_train, y_train)

            # Evaluate and log metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    main()
