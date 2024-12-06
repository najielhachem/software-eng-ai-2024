# src/ml_data_pipeline/config.py
from typing import Any, Dict

from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator


class DataLoaderConfig(BaseModel):
    """Configuration for the data loader.

    Attributes:
        file_path (str): The path to the data file.
        file_type (str): The type of the data file (csv or json).
    """

    file_path: str
    file_type: str

    @field_validator("file_type")
    def validate_file_type(cls, value: str) -> str:
        """Validates the file type.

        Args:
            value (str): The file type to validate.

        Returns:
            str: The validated file type.

        Raises:
            ValueError: If the file type is not 'csv' or 'json'.
        """
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value


class TransformationConfig(BaseModel):
    """Configuration for the data transformation.

    Attributes:
        normalize (bool): Whether to normalize the data.
        scaling_method (str): The method to use for scaling (standard or minmax).
    """

    normalize: bool
    scaling_method: str

    @field_validator("scaling_method")
    def validate_scaling_method(cls, value: str) -> str:
        """Validates the scaling method.

        Args:
            value (str): The scaling method to validate.

        Returns:
            str: The validated scaling method.

        Raises:
            ValueError: If the scaling method is not 'standard' or 'minmax'.
        """
        if value not in {"standard", "minmax"}:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        return value


class ModelConfig(BaseModel):
    """Configuration for the model.

    Attributes:
        type (str): The type of the model (linear or tree).
        params (Dict[str, Any]): Additional parameters for the model.
    """

    type: str
    params: Dict[str, Any] = {}

    @field_validator("type")
    def validate_model_type(cls, value: str) -> str:
        """Validates the model type.

        Args:
            value (str): The model type to validate.

        Returns:
            str: The validated model type.

        Raises:
            ValueError: If the model type is not 'linear' or 'tree'.
        """
        if value not in {"tree", "linear"}:
            raise ValueError("model type must be 'linear' or 'tree'")
        return value


class MLflowConfig(BaseModel):
    """Configuration for MLflow."""

    tracking_uri: str
    experiment_name: str


class Config(BaseModel):
    """Overall configuration for the pipeline.

    Attributes:
        data_loader (DataLoaderConfig): Configuration for the data loader.
        transformation (TransformationConfig): Configuration for the data transformation.
        model (ModelConfig): Configuration for the model.
        mlflow (MLflowConfig): Configuration for MLflow.
    """

    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig
    mlflow: MLflowConfig


def load_config(config_path: str) -> Config:
    """Loads the configuration from a file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Config: The loaded configuration.
    """
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config.model_validate(config_dict)
