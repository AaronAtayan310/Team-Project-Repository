# Team-Project-Repository
INST326 Team Project - Crime Research Data Pipeline

A comprehensive Python framework for automated processing, analysis, and visualization of crime datasets to support evidence-based policy making and criminal justice research.

# Team Members

Team Name: SAV 
| Name | Role | Responsibilities |
|------|------|------------------|
| [Aaron Atayan] | Project Lead | Project coordinator and designer |
| [Vyvyan Mai] | Visualizations Lead | Visualization function designer |
| [Sean McLean] | Functionality Tester | Developer and designer |

# Domain Focus and Problem Statement

Domain: Criminal justice and pattern analytics
Problem Statement: 
    Law enforcement agencies face problems regarding the processing and analyzing of crime data due to 
    data fragmentation, manual processing, and scalability issues.

# Installation and Setup Instructions for Project 1 Function Library

The crime research data pipeline function library created for Project 1 provides helper functions for **ingesting, cleaning, and analyzing crime rate data**. It forms the foundation of all other code in this repository, is written in Python, and relies on common data analysis libraries such as **pandas**. To find its source file in this repository, use `src` -> `proj1_crime_data_library.py`.

1. Clone or Download the Repository

If using Git:
```bash
git clone https://github.com/your-username/Team-Project-Repository.git
cd Team-Project-Repository
```
Or manually download the `proj1_crime_data_library.py` file and place it in your project directory.

2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. Install Required Dependencies

Create a `requirements.txt` file containing:
```
pandas>=2.0.0
```
Then install all requirements:
```bash
pip install -r requirements.txt
```

4. Import the Library in Your Script or Notebook

Once installed, you can import your function library as follows:

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

# Usage Examples of Key Functions in Project 1 Library

Below are practical examples of how to use the most important functions in the library.

1. Validate and Load a CSV File

```python
from proj1_crime_data_library import validate_csv_path, load_csv

file_path = "data/crime_data.csv"

if validate_csv_path(file_path):
    df = load_csv(file_path)
    print("Data successfully loaded!")
else:
    print("Invalid file path or format.")
```

2. Clean the Crime Dataset

```python
from proj1_crime_data_library import clean_crime_data

clean_df = clean_crime_data(df)
print(clean_df.head())
```

This function:

* Standardizes column names
* Removes duplicates
* Converts the `date` column to a datetime object
* Drops rows missing essential data

---

3. Calculate Missing Data Percentages

```python
from proj1_crime_data_library import calculate_missing_data

missing = calculate_missing_data(clean_df)
print("Missing data (%):")
print(missing)
```

Output:

```
crime_type       0.0
neighborhood     1.2
population       0.5
dtype: float64
```

---

4. Compute Annual Crime Rates

```python
from proj1_crime_data_library import compute_crime_rate_by_year

yearly_rates = compute_crime_rate_by_year(clean_df, population_col="population")
print(yearly_rates)
```

Example output:

```
   year  crime_count  population  crime_rate
0  2020         4235     8500000        49.8
1  2021         4170     8550000        48.7
```

---

5. Identify the Top Crime Types

```python
from proj1_crime_data_library import top_crime_types

top_crimes = top_crime_types(clean_df, n=5)
print(top_crimes)
```

Example output:

```
       crime_type  count
0         Theft     4123
1        Assault    2901
2    Burglary     1650
3    Robbery      1024
4    Vandalism     890
```

---

6. Find Areas with the Highest Crime Rates

```python
from proj1_crime_data_library import find_high_crime_areas

high_crime_areas = find_high_crime_areas(clean_df, area_col="neighborhood")
print(high_crime_areas.head(10))
```

Example output:

```
     neighborhood  crime_count
0    Downtown            532
1    Eastside           487
2    Midtown            453
```

---

7. Save Processed Data with Timestamped Filenames

```python
from proj1_crime_data_library import generate_timestamped_filename

filename = generate_timestamped_filename("cleaned_crime_data")
clean_df.to_csv(filename, index=False)
print(f"File saved as: {filename}")
```

Example output:

```
File saved as: cleaned_crime_data_2025-10-12_21-04-10.csv
```


# Overview and organization of Project 1 Function Library

There are, at time of writing, **21 total functions** contained within the crime research data pipeline library (Project 1). An important detail to take note of regarding all functions in the library (main function library file found in repo via `src` -> `proj1_crime_data_library.py`) is that the vast majority of them are designed to function with complete independence - that is, not relying on other functions in the library to be able to run. In terms of organization, the library file uses some comment lines (# ---) to group the 21 functions into 5 categories, with each category representing the common, overarching purpose in the project which that one group of functions (with each function achieving a more niche & specific goal) is intended to fulfill. These categories, and the names of all functions contained within these categories, are as follows...

**1. Data Ingestion**
- Contains the following 3 functions: load_csv, fetch_api_data, and validate_csv_path.
---
**2. Data Cleaning**
- Contains the following 5 functions: handle_missing_values, normalize_text_column, standardize_column_names, remove_outliers_iqr, and clean_crime_data.
---
**3. Data Transformation**
- Contains the following 2 functions: scale_features and generate_features.
---
**4. Data Analysis**
- Contains the following 7 functions: compute_summary_stats, run_regression, evaluate_model, calculate_missing_data, compute_crime_rate_by_year, top_crime_types, and find_high_crime_areas.
---
**5. Data Storage & Utilities**
- Contains the following 4 functions: save_to_csv, serialize_model, log_pipeline_step, and generate_timestamped_filename.

# Overview and organization of Project 2 Core Classes

The implementation of the crime research data pipeline core classes (Project 2) transforms the crime research data pipeline function library (Project 1) by defining **5 core classes**, all relevant in the context of the crime research data pipeline, such that each core class specifically represents the relevant integration of 1 out of the 5 categories of functions as defined in the function library (main function library file found in repo via `src` -> `proj1_crime_data_library.py`). To elaborate, in each core class many of its methods are intentionally **direct adaptations** of functions from the library category relevant to that core class, adjusted to work syntactically with object-oriented notation. An important detail to take note of regarding these core classes however is that **not every function from every function library category is adapted to be integrated into that category's relevant core class**, although the vast majority appear in these core classes. The organization of which core classes are intended to serve which overarching purposes in the crime research data pipeline as well as which methods from those core classes integrate relevant functions from the function library, is as follows...

**1. Ingesting Data - The DataIngestion Class (`src` -> `proj2_data_ingestion_cls`):**
- Intended to load crime data from various sources (CSV files, REST APis, etc) into the workspace of users of the project for later usage (research, analysis, etc).
- Methods load_csv(self, filepath: str), fetch_api_data(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) and @staticmethod validate_csv_path(file_path: str) integrate crime research data pipeline library functions load_csv, fetch_api_data, and validate_csv_path respectively.
---
**2. Cleaning Data - The DataCleaner Class (`src` -> `proj2_data_cleaning_cls`):**
- Intended to clean and preprocess pandas dataframes (the most common form of data for the sake of this project) to enhance the project users convenience of using these dataframes for later tasks (research, analysis, etc) in their workspace.
- Methods handle_missing_values(self, strategy: str = "mean", columns: Optional[List[str]] = None) and normalize_text_column(self, column: str, remove_special_chars: bool = False) integrate crime research data pipeline library functions handle_missing_values and normalize_text_column respectively.
---
**3. Transforming Data - The DataTransformation Class (`src` -> `proj2_data_transformation_cls`):**
- Intended to allow project users more simplified execution of very basic data transformation tasks on pandas dataframes (generating new features and scaling existing features).
- Methods scale_features(self, columns: list[str]) and generate_features(self) integrate crime research data pipeline library functions scale_features and generate_features respectively.
---
**4. Performing Analyses Of Data - The DataAnalysis Class (`src` -> `proj2_data_analysis_cls`):**
- Intended to allow project users to automate data analysis tasks on pandas dataframes that contain data of particular interest.
- Methods run_regression(self, y: pd.Series), evaluate_model(self, model: LinearRegression, y_test: pd.Series), calculate_missing_data(self), compute_crime_rate_by_year(self, population_col: str = "population"), top_crime_types(self, n: int = 10) and find_high_crime_areas(self, area_col: str = "neighborhood") integrate crime research data pipeline library functions run_regression, evaluate_model, calculate_missing_data, compute_crime_rate_by_year, top_crime_types, and find_high_crime_areas respectively.
---
**5. Storing Data & Extra Utilities - The DataStorageUtils Class (`src` -> `proj2_data_utilities_cls`):**
- Intended to automate general data pipeline operations such as serialization, logging, and file management, and also to allow project users to store relevant data in any of multiple formats (CSV, JSON, etc).
- Methods save_to_csv(self, df: pd.DataFrame, filepath: str, use_timestamp: bool = False, **kwargs), serialize_model(self, model: Any, path: str, metadata: Optional[Dict] = None), log_pipeline_step(self, step_name: str, status: str, extra_info: Optional[Dict] = None) and @staticmethod generate_timestamped_filename(base_name: str, extension: str = ".csv") integrate crime research data pipeline library functions save_to_csv, serialize_model, log_pipeline_step, and generate_timestamped_filename respectively.

# Overview and organization of Project 3 Refactored & Advanced Classes
The third implementation of the project features the most comprehensive and near-complete system yet, demonstrating advanced object-oriented programming principles including inheritance hierarchies, polymorphic behavior, abstract base classes, and composition relationships.

## System Architecture

### Inheritance Hierarchies

#### Processor Hierarchy
```
AbstractDataProcessor (Abstract Base Class)
├── NewDataAnalysis (statistical analysis & ML)
├── NewDataCleaner (missing values & text cleaning)
└── NewDataTransformation (scaling & feature engineering)
```

#### Source Hierarchy
```
AbstractDataSource (Abstract Base Class)
├── APIDataSource (REST API endpoints)
├── CSVDataSource (CSV file loading)
└── DatabaseDataSource (SQL database queries)
```

### Composition Relationships

- **DataPipeline** contains:
  - NewDataIngestion (has-a, data loading)
  - NewDataStorageUtils (has-a, file I/O)
  - Dynamic processors (has-many: cleaner/transformer/analyzer)

- **NewDataIngestion** contains:
  - data_sources[] (has-many, source tracking)
  - loaded_data{} (has-many, DataFrame cache)

## Key Features

### 1. Polymorphic Behavior
Same method calls produce different results based on object type:
- `process()` - Analysis vs cleaning vs transformation logic
- `load()` - API requests vs CSV reading vs SQL queries
- `validate()` - Data quality vs source availability vs connection validity
- `validate_source()` - URL format vs file existence vs DB credentials

### 2. Abstract Base Classes
Enforce consistent interfaces across implementations:
- **AbstractDataProcessor** - Requires process() and validate() methods
- **AbstractDataSource** - Requires load() and validate_source() methods

### 3. Composition Over Inheritance
- DataPipeline coordinates multiple specialized components
- NewDataIngestion delegates to polymorphic sources
- Fluent interface enables method chaining (load().clean().transform())

## Running the System
For demonstrations, refer to `tests_and_examples` -> `proj3_demonstration.py`.

This will comprehensively demonstrate:
- Inheritance hierarchies
- Polymorphic behavior
- Abstract base class enforcement
- Composition relationships
- Complete integrated system

## Run System Tests
For testing, refer to `tests_and_examples` -> `proj3_testing_suite.py`.

This will comprehensively demonstrate:
- Inheritance relationships
- Polymorphic method behavior
- Abstract class enforcement
- Composition functionality
- System integration

## Usage Examples

### Creating Data Sources
```python
from proj3_data_sources import APIDataSource, CSVDataSource, DatabaseDataSource

# Create different source types
api_source = APIDataSource("https://api.crime-data.com/v1/crimes")
csv_source = CSVDataSource("crime_data_2024.csv")
db_source = DatabaseDataSource("sqlite:///crime.db", "SELECT * FROM incidents")

# Polymorphic loading
df_api = api_source.load() # HTTP requests → JSON → DataFrame
df_csv = csv_source.load() # File validation → pd.read_csv()
df_db = db_source.load() # SQLAlchemy → pd.read_sql()
```

### Creating Processors
```python
from proj3_data_processors import NewDataAnalysis, NewDataCleaner, NewDataTransformation

# Create different processor types
analysis = NewDataAnalysis(df)
cleaner = NewDataCleaner(df, verbose=True)
transformer = NewDataTransformation(df)

# Polymorphic processing
analysis.process() # Statistical analysis + crime metrics
cleaner.process() # Missing value imputation
transformer.process() # Feature generation + scaling
```

### Managing a Complete Pipeline
```python
from proj3_data_pipeline import DataPipeline
from proj3_data_sources import CSVDataSource

# Create pipeline system
pipeline = DataPipeline(verbose=True)

# Load data polymorphically
source = CSVDataSource("crime_data.csv")
pipeline.load_data(source)

# Fluent pipeline execution
pipeline
.clean(strategy='mean')
.transform(scale_columns=['crime_count', 'population'])
.analyze()
.save_results("processed_crime_data.csv", use_timestamp=True)

# Get pipeline summary
summary = pipeline.get_summary()
print(f"Pipeline executed {summary['pipeline_steps']} steps")
```

## Design Decisions

### Why Inheritance for Processors and Sources?

**Processors** share common attributes (DataFrame, processing history) and operations (process, validate) but differ in:
- Processing logic (stats vs cleaning vs transformation)
- Validation criteria (data quality vs completeness vs transformability)
- Domain-specific analytics (crime rates, top types)

**Sources** share common interface (load, validate_source) but differ in:
- Data access mechanisms (HTTP vs file vs SQL)
- Validation requirements (URL vs path vs credentials)
- Metadata tracking (status codes vs file info vs query results)

Inheritance provides:
- Code reuse for shared DataFrame handling
- Polymorphic method implementation
- Clear "is-a" relationships

### Why Composition for DataPipeline?

**DataPipeline** coordinates multiple object types:
- Not "a type of" processor or source
- Needs flexible relationships with many components
- Should orchestrate, not inherit from, domain objects

**NewDataIngestion** manages source polymorphism:
- Delegates to any AbstractDataSource implementation
- Tracks loading history across source types
- Caches results without knowing source details

Composition provides:
- Flexibility to swap components
- Fluent method chaining interface
- Clear separation of concerns
- Dependency injection for testability

## Notable Outcomes

This implementation of the project demonstrates:

1. **Inheritance Design** - Logical hierarchies for data processing/loading
2. **Polymorphic Behavior** - Interchangeable sources and processors
3. **Abstract Interfaces** - Enforcing consistent data pipeline contracts
4. **Composition Patterns** - Flexible pipeline orchestration
5. **System Integration** - End-to-end crime data workflow

# Overview and organization of Project 4 Capstone Integration & Testing
The fourth implementation of the project features the most comprehensive and fully complete system yet. Project 4 is built upon all three previous projects, carrying over features from each project to create a full data pipeline. This project specifically includes a full demonstration with a sample dataset and a full test suite that covers each class and function. Our final project has the capabilities of data cleaning, transformation, and analysis. 

## Run System Tests

### Prerequisites

Before running tests, ensure you have:
- Python 3.x installed
- Required dependencies: see `proj4_final_requirements.txt` 
- All project modules in the `src/` directory

### Running All Tests

To run the complete test suite, run the file titled:
  `proj4_testing_suite.py`

This will execute all test classes and display detailed results for each test case.

There is also an included sample demonstration file titled:
  `proj4_demonstration.py`

### Running Specific Test Classes

The following Python unittest modules can be used if you want to run tests for a specific component:

```bash
# Test only DataSource enhancements
python -m unittest tests.test_proj4.TestEnhancedDataSource

# Test only DataProcessor enhancements
python -m unittest tests.test_proj4.TestEnhancedDataProcessor

# Test only PipelineManager
python -m unittest tests.test_proj4.TestPipelineManager
```

### Running Individual Tests

To run a single test method:

```bash
python -m unittest tests.test_proj4.TestEnhancedDataSource.test_uuid_identification
```

### Test Outputs

The test runner provides:
- **Verbose output** showing each test as it runs (pass/fail)
- **Summary statistics** including:
  - Total tests run
  - Number of successes
  - Number of failures
  - Number of errors
- **Detailed failure information** if any tests fail

### Test Coverage

The test suite covers:
- **Enhanced DataSource**: UUID identification, registry pattern, temporal tracking, load statistics, data lineage
- **Enhanced DataProcessor**: Performance monitoring, data quality tracking, original data preservation
- **DataQualityStandards**: Quality thresholds, field validation, freshness checks, completeness scoring
- **CrimeDataIngestion**: Factory methods, caching, ingestion statistics
- **CrimeDataStorageUtils**: File versioning, checksum verification, storage tracking
- **PipelineManager**: End-to-end pipeline orchestration, fluent interface, lineage tracking
- **Polymorphism**: Cross-component polymorphic behavior

### Exit Codes

- **0**: All tests passed successfully
- **1**: One or more tests failed


# Project contribution guidelines for team members
1. Ensure you have the latest code
2. Make your changes in the appropriate module
3. Write tests for new functionality
4. Update documentation as needed
5. Follow PEP 8 documentation guidelines
