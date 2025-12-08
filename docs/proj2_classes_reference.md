# Crime Research Data Pipeline Core Classes – Reference Guide

This document provides comprehensive reference information for all crime research data pipeline core classes (Project 2).

## Table of Contents

1. [Class I (`src -> proj2_data_ingestion_cls.py`) : dataIngestion](#the-dataingestion-class)
2. [Class II (`src -> proj2_data_cleaning_cls.py`) : dataCleaner](#the-datacleaner-class)
3. [Class III (`src -> proj2_data_transformation_cls.py`) : dataTransformation](#the-datatransformation-class)
4. [Class IV (`src -> proj2_data_analysis_cls.py`) : dataAnalysis](#the-dataanalysis-class)
5. [Class V (`src -> proj2_data_utilities_cls.py`) : dataStorageUtils](#the-datastorageutils-class)

---

## The dataIngestion class

### `__init__(default_timeout=10, track_sources=True)`

**Purpose:** Initialize an object of the dataIngestion class. Objects are intended to load crime data from various sources (CSV files, REST APis, etc) into the workspace of project users for later usage (research, analysis, etc). 

**Parameters:**
- `default_timeout` (int): Default timeout for API requests in seconds. Must be positive integer
- `track_sources` (bool): Whether to track uploaded sources

**Returns:** `dataIngestion` instance

**Raises:** `TypeError` if parameters are wrong types, `ValueError` if timeout ≤ 0

---

### `load_csv(filepath)`

**Purpose:** Load a CSV file into a pandas DataFrame.

**Parameters:**
- `filepath` (str): Path to the CSV file.

**Returns:** `pd.DataFrame` – Loaded data as a DataFrame

**Raises:** `TypeError` if filepath not string, `FileNotFoundError` if file doesn't exist

---

### `fetch_api_data(url, params=None, timeout=None)`

**Purpose:** Fetch JSON data from a REST API endpoint.

**Parameters:**
- `url` (str): The API URL
- `params` (dict, optional): Query parameters to include in request
- `timeout` (int, optional): Request timeout. Uses default_timeout if None

**Returns:** `dict` – Parsed JSON response

**Raises:** `TypeError` if URL not string or params not dict, `requests.RequestException` if request fails

---

### `@staticmethod validate_csv_path(file_path)`

**Purpose:** Validate whether a given file path points to an existing CSV file.

**Parameters:**
- `file_path` (str): The path to validate

**Returns:** `bool` – True if file exists and has `.csv` extension, False otherwise

**Raises:** `TypeError` if file_path not string

---

### `clear_sources()`

**Purpose:** Clears the list of tracked data sources.

**Parameters:** None

**Returns:** None

---

## The dataCleaner class

### `__init__(df, verbose=False)`

**Purpose:** Initialize an object of the dataCleaner class. Objects are intended to clean and preprocess pandas dataframes (the most common form of data for the sake of this project) to enhance the project users convenience of using these dataframes for later tasks (research, analysis, etc) in their workspace. 

**Parameters:**
- `df` (pd.DataFrame): The DataFrame to clean
- `verbose` (bool): Print information about cleaning operations

**Returns:** `dataCleaner` instance

**Raises:** `TypeError` if df not DataFrame, `ValueError` if df empty

---

### `handle_missing_values(strategy='mean', columns=None)`

**Purpose:** Handle missing values using specified strategy. Chainable method.

**Parameters:**
- `strategy` (str): 'mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill'
- `columns` (list[str], optional): Specific columns to apply strategy to

**Returns:** `dataCleaner` – Self for method chaining

**Raises:** `ValueError` if invalid strategy

---

### `normalize_text_column(column, remove_special_chars=False)`

**Purpose:** Normalize text in specified column (lowercase, strip, optional special char removal). Chainable.

**Parameters:**
- `column` (str): Column to normalize
- `remove_special_chars` (bool): Remove non-alphanumeric characters

**Returns:** `dataCleaner` – Self for method chaining

**Raises:** `ValueError` if column doesn't exist

---

## The dataTransformation class

### `__init__(frame)`

**Purpose:** Initialize an object of the dataTransformation class. Objects are intended to allow project users more simplified execution of very basic data transformation tasks on pandas dataframes (generating new features and scaling existing features).

**Parameters:**
- `frame` (pd.DataFrame): Input DataFrame for transformation

**Returns:** `dataTransformation` instance

**Raises:** `ValueError` if frame not DataFrame

---

### `scale_features(columns)`

**Purpose:** Scale specified numeric features using StandardScaler.

**Parameters:**
- `columns` (list[str]): Columns to scale

**Returns:** None (modifies internal frame)

---

### `generate_features()`

**Purpose:** Generate new derived features (value_per_count ratio if value/count columns exist).

**Parameters:** None

**Returns:** None (modifies internal frame)

---

## The dataAnalysis class

### `__init__(frame)`

**Purpose:** Initialize an object of the dataAnalysis class. Objects are intended to allow project users to automate data analysis tasks on pandas dataframes that contain data of particular interest.

**Parameters:**
- `frame` (pd.DataFrame): DataFrame containing data to analyze

**Returns:** `dataAnalysis` instance

**Raises:** `ValueError` if frame not DataFrame

---

### `run_regression(y)`

**Purpose:** Fit a simple linear regression model.

**Parameters:**
- `y` (pd.Series): Target variable

**Returns:** `LinearRegression` – Trained model

---

### `evaluate_model(model, y_test)`

**Purpose:** Evaluate trained regression model using Mean Squared Error.

**Parameters:**
- `model` (LinearRegression): Trained model
- `y_test` (pd.Series): Test targets

**Returns:** `dict` – `{'mse': float}`

---

### `calculate_missing_data()`

**Purpose:** Calculate percentage of missing data in each column.

**Parameters:** None

**Returns:** `pd.Series` – Missing data percentages per column

---

### `compute_crime_rate_by_year(population_col='population')`

**Purpose:** Compute annual crime rates per 100,000 people.

**Parameters:**
- `population_col` (str): Population column name

**Returns:** `pd.DataFrame` – Columns: `['year', 'crime_count', 'crime_rate']`

**Raises:** `ValueError` if required columns missing

---

### `top_crime_types(n=10)`

**Purpose:** Identify top N most frequent crime types.

**Parameters:**
- `n` (int): Number of top types

**Returns:** `pd.DataFrame` – Columns: `['crime_type', 'count']`

**Raises:** `ValueError` if 'crime_type' column missing

---

### `find_high_crime_areas(area_col='neighborhood')`

**Purpose:** Identify areas with highest crime counts.

**Parameters:**
- `area_col` (str): Geographic area column name

**Returns:** `pd.DataFrame` – Areas sorted by descending crime count

**Raises:** `ValueError` if area column missing

---

## The dataStorageUtils class

### `__init__(base_output_dir=None, log_level=logging.INFO)`

**Purpose:** Initialize an object of the dataStorageUtils class. Objects are intended to automate general data pipeline operations such as serialization, logging, and file management, and also to allow project users to store relevant data in any of multiple formats (CSV, JSON, etc).

**Parameters:**
- `base_output_dir` (str, optional): Base directory for outputs
- `log_level` (int): Logging level

**Returns:** `dataStorageUtils` instance

---

### `save_to_csv(df, filepath, use_timestamp=False, **kwargs)`

**Purpose:** Save DataFrame to CSV file (optional timestamp).

**Parameters:**
- `df` (pd.DataFrame): DataFrame to save
- `filepath` (str): Destination path
- `use_timestamp` (bool): Add timestamp to filename
- `**kwargs`: Passed to `pd.to_csv()`

**Returns:** `Path` – Actual save path

---

### `load_from_csv(filepath, **kwargs)`

**Purpose:** Load DataFrame from CSV file.

**Parameters:**
- `filepath` (str): Source file path
- `**kwargs`: Passed to `pd.read_csv()`

**Returns:** `pd.DataFrame`

**Raises:** `FileNotFoundError` if file missing

---

### `serialize_model(model, path, metadata=None)`

**Purpose:** Serialize model to pickle file (optional JSON metadata).

**Parameters:**
- `model` (Any): Model to save
- `path` (str): Save path
- `metadata` (dict, optional): Metadata to save as JSON

**Returns:** `Path` – Save path

---

### `deserialize_model(path)`

**Purpose:** Load model from pickle file.

**Parameters:**
- `path` (str): File path

**Returns:** Model object

**Raises:** `FileNotFoundError` if file missing

---

### `save_to_json(data, filepath, use_timestamp=False, **kwargs)`

**Purpose:** Save data to JSON file.

**Parameters:**
- `data` (dict/list): Data to save
- `filepath` (str): Destination path
- `use_timestamp` (bool): Add timestamp
- `**kwargs`: Passed to `json.dump()`

**Returns:** `Path`

---

### `log_pipeline_step(step_name, status, extra_info=None)`

**Purpose:** Log pipeline step with status and optional info.

**Parameters:**
- `step_name` (str): Step name
- `status` (str): 'started', 'completed', 'failed', etc.
- `extra_info` (dict, optional): Additional info

**Returns:** None

---

### `@staticmethod generate_timestamped_filename(base_name, extension='.csv')`

**Purpose:** Generate timestamped filename (format: `YYYY-MM-DD_HH-MM-SS`).

**Parameters:**
- `base_name` (str): Base filename
- `extension` (str): File extension

**Returns:** `str` – Generated filename

**Raises:** `TypeError` if parameters not strings

---

### `compute_file_hash(filepath, algorithm='sha256')`

**Purpose:** Compute file hash for integrity checking.

**Parameters:**
- `filepath` (str): File path
- `algorithm` (str): Hash algorithm

**Returns:** `str` – Hex hash

**Raises:** `FileNotFoundError`

---

### `create_pipeline_manifest(manifest_data, filepath=None)`

**Purpose:** Create JSON manifest documenting pipeline execution.

**Parameters:**
- `manifest_data` (dict): Pipeline info
- `filepath` (str, optional): Custom manifest path

**Returns:** `Path`

---

### `get_directory_size(dirpath)`

**Purpose:** Calculate total size of directory in bytes.

**Parameters:**
- `dirpath` (str): Directory path

**Returns:** `int` – Total bytes

**Raises:** `FileNotFoundError`

---
