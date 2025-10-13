import os
import json
import logging
import pickle
import requests
import pandas as pd
from datetime import datetime
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

def validate_csv_path(file_path: str) -> bool:
    """
    Validate whether a given file path points to an existing CSV file.

    Args:
        file_path (str): The path to the file being validated.

    Returns:
        bool: True if the file exists and has a '.csv' extension, False otherwise.

    Raises:
        TypeError: If 'file_path' is not a string.
    """
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string")

    return os.path.isfile(file_path) and file_path.lower().endswith(".csv")


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

def standardize_column_names(columns: list[str]) -> list[str]:
    """
    Standardize column names by stripping whitespace, converting to lowercase,
    and replacing spaces with underscores.

    Args:
        columns (list[str]): A list of column names.

    Returns:
        list[str]: A list of standardized column names.

    Raises:
        TypeError: If 'columns' is not a list of strings.
    """
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise TypeError("Columns must be a list of strings")

    return [col.strip().lower().replace(" ", "_") for col in columns]


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.

    Outliers are defined as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to filter for outliers.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.

    Raises:
        TypeError: If 'df' is not a pandas DataFrame.
        ValueError: If 'column' does not exist in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


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

def calculate_missing_data(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage of missing data in each column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.Series: A Series containing the percentage of missing data per column.

    Raises:
        TypeError: If 'df' is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    return (df.isnull().sum() / len(df)) * 100


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