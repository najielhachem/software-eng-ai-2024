from __future__ import annotations

import loguru
import pandas as pd

from ml_data_pipeline.data_transformer.base_transformer import DataTransformer
from ml_data_pipeline.models.base_model import Model


class InferencePipeline:
    _logger: loguru.Logger
    _data_transformer: DataTransformer
    _model: Model

    def __init__(
        self, logger: loguru.Logger, data_transformer: DataTransformer, model: Model
    ):
        self._logger = logger
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
            self._logger.info("Pipeline execution started.")

            self._logger.info("Applying Data transformation.")
            transformed_data = self._data_transformer.transform(data)
            self._logger.debug(f"Data: {transformed_data.head()}")
            self._logger.info("Data transformed successfully.")

            self._logger.info("Running Inference.")
            predictions = self._model.predict(transformed_data)
            self._logger.debug(f"Predictions: {predictions.head()}")
            self._logger.info("Model training and prediction completed successfully.")

            self._logger.info("Pipeline execution completed.")

        except Exception as e:
            self._logger.error(f"Failed in Pipeline Execution: {e}")
            return
