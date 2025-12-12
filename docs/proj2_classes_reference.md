# Crime Research Data Pipeline Core Classes – Reference Guide

This document provides comprehensive reference information for all crime research data pipeline core classes (Project 2).

## Table of Contents

1. [Class I (`src -> proj2_data_ingestion_cls.py`) : DataIngestion](#the-dataingestion-class)
2. [Class II (`src -> proj2_data_cleaning_cls.py`) : DataCleaner](#the-datacleaner-class)
3. [Class III (`src -> proj2_data_transformation_cls.py`) : DataTransformation](#the-datatransformation-class)
4. [Class IV (`src -> proj2_data_analysis_cls.py`) : DataAnalysis](#the-dataanalysis-class)
5. [Class V (`src -> proj2_data_utilities_cls.py`) : DataStorageUtils](#the-datastorageutils-class)

---

## The DataIngestion Class

### **Properties**

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`default_timeout`** | `int` | Read/Write | Default API request timeout in seconds. |
| **`track_sources`** | `bool` | Read/Write | Tracks whether data source logging is active. |
| **`data_sources`** | `list` | Read-only | List of tracked data sources (URLs, file paths, etc.). |

### `__init__(default_timeout=10, track_sources=True)`

**Purpose:** Initialize an object of the DataIngestion class. Objects load crime data from various sources (CSV, APIs).

**Parameters:**
- `default_timeout` (int): Default timeout for API requests in seconds. Must be a positive integer.
- `track_sources` (bool): Whether to track uploaded sources.

**Returns:** `DataIngestion` object / instance

**Raises:** `TypeError` if parameters are wrong types, `ValueError` if timeout $\leq 0$.

---

### `load_csv(filepath)`

**Purpose:** Load a CSV file into a pandas DataFrame and track the source if enabled.

**Parameters:**
- `filepath` (str): Path to the CSV file.

**Returns:** `pd.DataFrame` – Loaded data as a DataFrame.

**Raises:** `TypeError` if filepath not string, `FileNotFoundError` if file doesn't exist.

---

### `fetch_api_data(url, params=None, timeout=None)`

**Purpose:** Fetch JSON data from a REST API endpoint and track the source if enabled.

**Parameters:**
- `url` (str): The API URL.
- `params` (dict, optional): Query parameters to include in request.
- `timeout` (int, optional): Request timeout. Uses `default_timeout` if `None`.

**Returns:** `Union[dict, list]` – Parsed JSON response (list or dictionary).

**Raises:** `TypeError` if URL not string or params not dict, `requests.RequestException` if request fails (e.g., connection error or bad status code).

---

### `@staticmethod validate_csv_path(file_path)`

**Purpose:** Validate whether a given file path points to an existing CSV file.

**Parameters:**
- `file_path` (str): The path to validate.

**Returns:** `bool` – True if file exists and has `.csv` extension, False otherwise.

**Raises:** `TypeError` if `file_path` not string.

---

### `clear_sources()`

**Purpose:** Clears the list of tracked data sources.

**Parameters:** None

**Returns:** None

---

## The DataCleaner Class

### **Properties**

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`df`** | `pd.DataFrame` | Read/Write | The DataFrame currently held and being cleaned (Public Property). |
| **`verbose`** | `bool` | Read/Write | Controls whether cleaning operations are printed to console. |
| **`cleaning_history`** | `list` | Read-only | Log of all cleaning operations performed. |

### `__init__(df, verbose=False)`

**Purpose:** Initialize an object of the DataCleaner class. Objects clean and preprocess pandas DataFrames to enhance usability for later tasks.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame to clean.
- `verbose` (bool): Print information about cleaning operations.

**Returns:** `DataCleaner` object / instance

**Raises:** `TypeError` if `df` not DataFrame, `ValueError` if `df` is empty.

---

### `handle_missing_values(strategy='mean', columns=None)`

**Purpose:** Handle missing values using specified strategy. Chainable method.

**Parameters:**
- `strategy` (str): `'mean'`, `'median'`, `'mode'`, `'drop'`, `'forward_fill'`, `'backward_fill'`.
- `columns` (List[str], optional): Specific columns to apply strategy to. If `None`, applies to all compatible columns.

**Returns:** `DataCleaner` – Self for method chaining.

**Raises:** `ValueError` if invalid strategy is provided.

---

### `normalize_text_column(column, remove_special_chars=False)`

**Purpose:** Normalize text in specified column (lowercase, strip whitespace, optional special char removal). Chainable.

**Parameters:**
- `column` (str): Column to normalize.
- `remove_special_chars` (bool): If `True`, removes non-alphanumeric characters.

**Returns:** `DataCleaner` – Self for method chaining.

**Raises:** `KeyError` if column doesn't exist.

---

## The DataTransformation Class

### **Properties**

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`frame`** | `pd.DataFrame` | Read/Write | The DataFrame currently held for transformation (Public Property). |

### `__init__(frame)`

**Purpose:** Initialize an object of the DataTransformation class to simplify execution of basic data transformation tasks (generating new features and scaling existing features).

**Parameters:**
- `frame` (pd.DataFrame): Input DataFrame for transformation.

**Returns:** `DataTransformation` object / instance

**Raises:** `TypeError` if `frame` not DataFrame.

---

### `scale_features(columns)`

**Purpose:** Scale specified numeric features using `StandardScaler` (Z-score scaling).

**Parameters:**
- `columns` (List[str]): Columns to scale. Must contain numeric data.

**Returns:** None (modifies internal `frame` property).

**Raises:** `KeyError` if any column is not found in the DataFrame.

---

### `generate_features()`

**Purpose:** Generate new derived features, specifically `value_per_count` ratio if the `value` and `count` columns exist.

**Parameters:** None

**Returns:** None (modifies internal `frame` property).

---

## The DataAnalysis Class

### **Properties**

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`frame`** | `pd.DataFrame` | Read/Write | The DataFrame containing data for analysis (Public Property). |
| **`described`** | `pd.DataFrame` | Read-only | Descriptive statistics of the numeric columns in the current `frame`. |

### `__init__(frame)`

**Purpose:** Initialize an object of the DataAnalysis class to automate data analysis tasks on pandas DataFrames.

**Parameters:**
- `frame` (pd.DataFrame): DataFrame containing data to analyze.

**Returns:** `DataAnalysis` object / instance

**Raises:** `TypeError` if `frame` not DataFrame.

---

### `run_regression(y)`

**Purpose:** Fit a simple linear regression model using all numeric columns in `frame` (excluding `y`) as predictors.

**Parameters:**
- `y` (pd.Series): Target variable for the regression.

**Returns:** `LinearRegression` – Trained model object.

---

### `evaluate_model(model, y_test)`

**Purpose:** Evaluate trained regression model using Mean Squared Error (MSE).

**Parameters:**
- `model` (LinearRegression): Trained model object.
- `y_test` (pd.Series): True target values to compare against.

**Returns:** `dict` – `{'mse': float}`

---

### `calculate_missing_data()`

**Purpose:** Calculate percentage of missing data in each column of the `frame`.

**Parameters:** None

**Returns:** `pd.Series` – Missing data percentages per column, indexed by column name.

---

### `compute_crime_rate_by_year(population_col='population')`

**Purpose:** Compute annual crime rates per 100,000 people, requiring a 'date' column.

**Parameters:**
- `population_col` (str): Population column name in the DataFrame.

**Returns:** `pd.DataFrame` – Columns: `['year', 'crime_count', 'crime_rate']`.

**Raises:** `KeyError` if required columns (`date`, `population_col`) are missing.

---

### `top_crime_types(n=10)`

**Purpose:** Identify top N most frequent crime types based on the 'crime\_type' column.

**Parameters:**
- `n` (int): Number of top types to return.

**Returns:** `pd.DataFrame` – Columns: `['crime_type', 'count']`.

**Raises:** `KeyError` if 'crime\_type' column is missing.

---

### `find_high_crime_areas(area_col='neighborhood')`

**Purpose:** Identify areas with the highest crime counts, aggregated by the specified area column.

**Parameters:**
- `area_col` (str): Geographic area column name.

**Returns:** `pd.DataFrame` – Areas sorted by descending crime count.

**Raises:** `KeyError` if the specified area column is missing.

---

## The DataStorageUtils Class

### **Properties**

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`base_output_dir`** | `pathlib.Path` | Read/Write | The base directory where all generated files are saved. |
| **`log_level`** | `int` | Read-only | The current minimum level for logging messages (e.g., `logging.INFO`). |

### `__init__(base_output_dir=None, log_level=logging.INFO)`

**Purpose:** Initialize an object for automating pipeline operations (serialization, logging, and file management).

**Parameters:**
- `base_output_dir` (str, optional): Base directory for outputs. Defaults to current working directory (`Path.cwd()`).
- `log_level` (int): Logging level (e.g., `logging.INFO`, `20`).

**Returns:** `DataStorageUtils` object / instance

**Raises:** `TypeError` if `log_level` is not an integer.

---

### `save_to_csv(df, filepath, use_timestamp=False, **kwargs)`

**Purpose:** Save DataFrame to CSV file (optional timestamp).

**Parameters:**
- `df` (pd.DataFrame): DataFrame to save.
- `filepath` (str): Destination path.
- `use_timestamp` (bool): Add timestamp to filename.
- `**kwargs`: Additional arguments passed to `pd.DataFrame.to_csv()`.

**Returns:** `pathlib.Path` – Actual save path.

**Raises:** `TypeError` if `df` is not a DataFrame.

---

### `load_from_csv(filepath, **kwargs)`

**Purpose:** Load DataFrame from CSV file.

**Parameters:**
- `filepath` (str): Source file path.
- `**kwargs`: Additional arguments passed to `pd.read_csv()`.

**Returns:** `pd.DataFrame`

**Raises:** `FileNotFoundError` if file missing.

---

### `serialize_model(model, path, metadata=None)`

**Purpose:** Serialize (save) a model object to disk using pickle (optional JSON metadata saved alongside).

**Parameters:**
- `model` (Any): Model to save.
- `path` (str): File path to save the model.
- `metadata` (dict, optional): Descriptive metadata to save as JSON.

**Returns:** `pathlib.Path` – Save path.

---

### `deserialize_model(path)`

**Purpose:** Deserialize (load) a model object from a pickle file.

**Parameters:**
- `path` (str): File path.

**Returns:** Model object (`Any`)

**Raises:** `FileNotFoundError` if file missing.

---

### `save_to_json(data, filepath, use_timestamp=False, **kwargs)`

**Purpose:** Save data (dict or list) to JSON file.

**Parameters:**
- `data` (Union[dict, list]): Data to save.
- `filepath` (str): Destination path.
- `use_timestamp` (bool): Add timestamp.
- `**kwargs`: Additional arguments passed to `json.dump()`.

**Returns:** `pathlib.Path`

**Raises:** `TypeError` if data is not a dict or list.

---

### `log_pipeline_step(step_name, status, extra_info=None)`

**Purpose:** Log pipeline step using the configured logger.

**Parameters:**
- `step_name` (str): Name of the step.
- `status` (str): Status message (e.g., `'started'`, `'completed'`, `'failed'`).
- `extra_info` (dict, optional): Additional information to log.

**Returns:** None

---

### `@staticmethod generate_timestamped_filename(base_name, extension='.csv')`

**Purpose:** Generate timestamped filename (format: `base_name\_YYYY-MM-DD\_HH-MM-SS.ext`).

**Parameters:**
- `base_name` (str): Base filename (without extension).
- `extension` (str): File extension (e.g., `'.json'`, `'.csv'`).

**Returns:** `str` – Generated filename.

**Raises:** `TypeError` if parameters not strings.

---

### `compute_file_hash(filepath, algorithm='sha256')`

**Purpose:** Compute hash of a file for integrity checking.

**Parameters:**
- `filepath` (str): File path.
- `algorithm` (str): Hash algorithm to use.

**Returns:** `str` – Hexadecimal hash string.

**Raises:** `FileNotFoundError`, `ValueError` if the hash algorithm is unsupported.

---

### `create_pipeline_manifest(manifest_data, filepath=None)`

**Purpose:** Create JSON manifest documenting pipeline execution, automatically adding a `created_at` timestamp.

**Parameters:**
- `manifest_data` (dict): Pipeline information (e.g., steps, file paths).
- `filepath` (str, optional): Custom manifest path. If `None`, a timestamped file is used in `base_output_dir`.

**Returns:** `pathlib.Path` – Path where manifest was saved.

---

### `get_directory_size(dirpath)`

**Purpose:** Calculate total size of a directory in bytes.

**Parameters:**
- `dirpath` (str): Directory path.

**Returns:** `int` – Total size in bytes.

**Raises:** `FileNotFoundError`, `NotADirectoryError` if the path exists but is not a directory.

---
