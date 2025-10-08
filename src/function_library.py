import os
import json
import logging
import pickle
import requests
import pandas as pd
from typing import Any, Dict, List, Iterator, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    return pd.read_csv(filepath)


def fetch_api_data(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetch JSON data from a REST API endpoint.

    Args:
        url (str): The API URL.
        params (dict, optional): Query parameters to include in the request.

    Returns:
        dict: Parsed JSON response.

    Raises:
        requests.RequestException: If the API request fails.
    """
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in a DataFrame using a given strategy.

    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Method to handle missing values ('mean', 'median', 'drop').

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if strategy == "mean":
        cleaned_df.fillna(cleaned_df.mean(numeric_only=True), inplace=True)
    elif strategy == "median":
        cleaned_df.fillna(cleaned_df.median(numeric_only=True), inplace=True)
    elif strategy == "drop":
        cleaned_df.dropna(inplace=True)
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'drop'.")

    return cleaned_df


def normalize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize text in a specified column by lowercasing and stripping spaces.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized text column.
    """
    df = df.copy()
    df[column] = df[column].astype(str).str.lower().str.strip()
    return df


# ---------------------------------------------------------------------------
# 3. DATA TRANSFORMATION
# ---------------------------------------------------------------------------

def scale_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Scale numeric features using standardization.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str]): Columns to scale.

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new derived features.

    Example: Create a ratio between two numeric columns if they exist.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    df = df.copy()

    if "value" in df.columns and "count" in df.columns:
        df["value_per_count"] = df["value"] / (df["count"] + 1e-9)

    return df


# ---------------------------------------------------------------------------
# 4. DATA ANALYSIS / MODELING
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic summary statistics for numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics.
    """
    return df.describe()


def run_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """
    Fit a simple linear regression model.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        LinearRegression: Trained regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a trained regression model using Mean Squared Error (MSE).

    Args:
        model (LinearRegression): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {"mse": mse}


# ---------------------------------------------------------------------------
# 5. DATA STORAGE
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. UTILITIES / ORCHESTRATION
# ---------------------------------------------------------------------------

def log_pipeline_step(step_name: str, status: str) -> None:
    """
    Log a pipeline step for monitoring purposes.

    Args:
        step_name (str): Name of the step.
        status (str): Status message (e.g., 'started', 'completed', 'failed').
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Step '{step_name}' - Status: {status}")


def run_pipeline(config: Dict[str, Any]) -> None:
    """
    Execute a simple data pipeline based on configuration.

    Args:
        config (dict): Pipeline configuration dictionary.
    """
    log_pipeline_step("Pipeline", "started")

    # Example: Load data, clean it, compute stats, and save results
    df = load_csv(config["input_path"])
    df = handle_missing_values(df)
    df = generate_features(df)

    stats = compute_summary_stats(df)
    save_to_csv(stats, config["output_path"])

    log_pipeline_step("Pipeline", "completed")