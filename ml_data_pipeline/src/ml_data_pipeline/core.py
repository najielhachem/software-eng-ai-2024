from __future__ import annotations

import pandas as pd
from loguru import logger

from ml_data_pipeline.config import ModelConfig, TransformationConfig
from ml_data_pipeline.data_transformer.base_transformer import DataTransformer
from ml_data_pipeline.data_transformer.factory import TransformerFactory
from ml_data_pipeline.models.base_model import Model
from ml_data_pipeline.models.factory import ModelFactory


def load_pipeline(
    transformation_config: TransformationConfig, model_config: ModelConfig
) -> "InferencePipeline":
    data_transformer = TransformerFactory.get_transformer(
        transformation_config.scaling_method
    )
    model = ModelFactory.get_model(model_config)
    return InferencePipeline(data_transformer, model)


class InferencePipeline:
    _data_transformer: DataTransformer
    _model: Model

    def __init__(self, data_transformer: DataTransformer, model: Model):
        self._data_transformer = data_transformer
        self._model = model

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs the inference data pipeline on the input data.

        Args:
            data (pd.DataFrame): The input data to process.

        Returns:
            pd.DataFrame: The processed data.
        """

        try:
            logger.info("Pipeline execution started.")

            logger.info("Applying Data transformation.")
            transformed_data = self._data_transformer.transform(data)
            logger.debug(f"Data: {transformed_data.head()}")
            logger.info("Data transformed successfully.")

            logger.info("Running Inference.")
            predictions = self._model.predict(transformed_data)
            logger.debug(f"Predictions: {predictions.head()}")
            logger.info("Model prediction completed successfully.")

            logger.info("Pipeline execution completed.")
            return predictions

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            return
