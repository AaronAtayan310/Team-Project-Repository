# Crime Research Data Pipeline Function Library – Reference Guide

This document provides comprehensive reference information for all functions in proj1_crime_data_library.py, the crime research data pipeline library (Project 1).

## Table of Contents

1. [Data Ingestion Functions](#data-ingestion-functions)
2. [Data Cleaning Functions](#data-cleaning-functions)
3. [Data Transformation Functions](#data-transformation-functions)
4. [Data Analysis Functions](#data-analysis-functions)
5. [Data Storage & Utilities Functions](#data-storage--utilities-functions)

---

## Data Ingestion Functions

### load_csv(filepath)

**Purpose:** Load a CSV file into a pandas DataFrame.

**Parameters:**
- `filepath` (str): Path to the CSV file.

**Returns:** `pd.DataFrame` – Loaded data as a DataFrame

**Raises:** `FileNotFoundError` if the file does not exist

---

### fetch_api_data(url, params=None)

**Purpose:** Fetch JSON data from a REST API endpoint.

**Parameters:**
- `url` (str): The API URL
- `params` (dict, optional): Query parameters for the request

**Returns:** `dict` – Parsed JSON response

**Raises:** `requests.RequestException` if the API request fails

---

### validate_csv_path(file_path)

**Purpose:** Validate whether a given file path points to an existing CSV file.

**Parameters:**
- `file_path` (str): The path to validate

**Returns:** `bool` – True if file exists and has a `.csv` extension, False otherwise

**Raises:** `TypeError` if the input is not a string

---

## Data Cleaning Functions

### handle_missing_values(df, strategy='mean')

**Purpose:** Handle missing values in a DataFrame using a specified strategy.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `strategy` (str): Method ('mean', 'median', 'drop')

**Returns:** `pd.DataFrame` – Cleaned DataFrame

**Raises:** `ValueError` if the strategy is invalid

---

### normalize_text_column(df, column)

**Purpose:** Normalize text in a specified column by lowercasing and stripping spaces.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `column` (str): Column to normalize

**Returns:** `pd.DataFrame` – DataFrame with normalized text column

---

### standardize_column_names(columns)

**Purpose:** Standardize column names by stripping whitespace, converting to lowercase, and replacing spaces with underscores.

**Parameters:**
- `columns` (list of str): List of column names

**Returns:** `list[str]` – Standardized column names

**Raises:** `TypeError` if columns is not a list of strings

---

### remove_outliers_iqr(df, column)

**Purpose:** Remove outliers from a column using the Interquartile Range (IQR) method.

**Parameters:**
- `df` (pd.DataFrame): The input DataFrame
- `column` (str): Column to process

**Returns:** `pd.DataFrame` – DataFrame with outliers removed

**Raises:** `TypeError` if `df` is not a DataFrame  
**Raises:** `ValueError` if `column` is not found

---

### clean_crime_data(df)

**Purpose:** Perform general cleaning on a crime dataset. Steps:
- Standardize column names
- Drop duplicate rows
- Remove records with missing essential values
- Convert date columns to datetime

**Parameters:**
- `df` (pd.DataFrame): Raw crime dataset

**Returns:** `pd.DataFrame` – Cleaned DataFrame

---

## Data Transformation Functions

### scale_features(df, columns)

**Purpose:** Scale numeric features using standardization.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `columns` (list of str): Columns to scale

**Returns:** `pd.DataFrame` – DataFrame with scaled features

---

### generate_features(df)

**Purpose:** Generate new derived features (e.g., create a ratio between two numeric columns if they exist).

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:** `pd.DataFrame` – DataFrame with additional features

---

## Data Analysis Functions

### compute_summary_stats(df)

**Purpose:** Compute basic summary statistics for numeric columns.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:** `pd.DataFrame` – Descriptive statistics

---

### run_regression(X, y)

**Purpose:** Fit a simple linear regression model.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target variable

**Returns:** `LinearRegression` – Trained regression model

---

### evaluate_model(model, X_test, y_test)

**Purpose:** Evaluate a trained regression model using Mean Squared Error (MSE).

**Parameters:**
- `model` (LinearRegression): Trained model
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series): Test targets

**Returns:** `dict` – Dictionary with evaluation metrics (`mse`)

---

### calculate_missing_data(df)

**Purpose:** Calculate the percentage of missing data in each column.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze

**Returns:** `pd.Series` – Percentage of missing data per column

**Raises:** `TypeError` if input is not a DataFrame

---

### compute_crime_rate_by_year(df, population_col='population')

**Purpose:** Compute annual crime rates per 100,000 people.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with 'date' and 'population' columns
- `population_col` (str): Population column name

**Returns:** `pd.DataFrame` – DataFrame with columns `year`, `crime_count`, `crime_rate`

**Raises:** `ValueError` if required columns are missing

---

### top_crime_types(df, n=10)

**Purpose:** Identify the top N most frequent crime types.

**Parameters:**
- `df` (pd.DataFrame): Dataset with a 'crime_type' column
- `n` (int): Number of top types to return

**Returns:** `pd.DataFrame` – DataFrame with columns `crime_type`, `count`

**Raises:** `ValueError` if the required column is missing

---

### find_high_crime_areas(df, area_col='neighborhood')

**Purpose:** Identify areas with the highest number of reported crimes.

**Parameters:**
- `df` (pd.DataFrame): Dataset with area column
- `area_col` (str): Name of geographic area column

**Returns:** `pd.DataFrame` – DataFrame of areas sorted by descending crime count

**Raises:** `ValueError` if the area column is missing

---

## Data Storage & Utilities Functions

### save_to_csv(df, filepath)

**Purpose:** Save a DataFrame to a CSV file.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to save
- `filepath` (str): Destination file path

**Returns:** None

---

### serialize_model(model, path)

**Purpose:** Serialize (save) a model object to disk using pickle.

**Parameters:**
- `model` (Any): Trained model
- `path` (str): File path for saving

**Returns:** None

---

### log_pipeline_step(step_name, status)

**Purpose:** Log a pipeline step for monitoring purposes.

**Parameters:**
- `step_name` (str): Name of the step
- `status` (str): Status message (e.g., 'started', 'completed', 'failed')

**Returns:** None

---

### generate_timestamped_filename(base_name, extension='.csv')

**Purpose:** Generate a timestamped filename with a given base name and extension (format: `YYYY-MM-DD_HH-MM-SS`).

**Parameters:**
- `base_name` (str): Base name of the file
- `extension` (str, optional): File extension (default: '.csv')

**Returns:** `str` – The generated filename

**Raises:** `TypeError` if base_name or extension is not a string

---
