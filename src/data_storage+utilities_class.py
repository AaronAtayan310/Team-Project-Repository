import logging
import pickle
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Iterator, Optional

def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Destination file path.
    """
    df.to_csv(filepath, index=False)


def serialize_model(model: Any, path: str) -> None:
    """
    Serialize (save) a model object to disk using pickle.

    Args:
        model (Any): Trained model.
        path (str): File path to save the model.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)

def log_pipeline_step(step_name: str, status: str) -> None:
    """
    Log a pipeline step for monitoring purposes.

    Args:
        step_name (str): Name of the step.
        status (str): Status message (e.g., 'started', 'completed', 'failed').
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Step '{step_name}' - Status: {status}")

def generate_timestamped_filename(base_name: str, extension: str = ".csv") -> str:
    """
    Generate a timestamped filename with a given base name and extension.

    The timestamp follows the format: YYYY-MM-DD_HH-MM-SS.

    Args:
        base_name (str): The base name of the file (without extension).
        extension (str, optional): The file extension to append. Defaults to '.csv'.

    Returns:
        str: The generated filename including the timestamp and extension.

    Raises:
        TypeError: If 'base_name' or 'extension' is not a string.
    """
    if not isinstance(base_name, str) or not isinstance(extension, str):
        raise TypeError("Base name and extension must be strings")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}{extension}"
