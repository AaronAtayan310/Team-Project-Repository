# Team-Project-Repository
INST326 Team Project - Data Pipeline Framework for Crime rates/datasets

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

# Installation and Setup Instructions

This library provides helper functions for **ingesting, cleaning, and analyzing crime rate data**. It is written in Python and relies on common data analysis libraries such as **pandas**.

1. Clone or Download the Repository

If using Git:
```bash
git clone https://github.com/yourusername/crime-data-library.git
cd crime-data-library
```
Or manually download the `.py` file and place it in your project directory.

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

# Usage Examples for Key Functions

Below are practical examples of how to use the most important functions in the library.

1. Validate and Load a CSV File

```python
from crime_data_library import validate_csv_path, load_crime_data

file_path = "data/crime_data.csv"

if validate_csv_path(file_path):
    df = load_crime_data(file_path)
    print("Data successfully loaded!")
else:
    print("Invalid file path or format.")
```

2. Clean the Crime Dataset

```python
from crime_data_library import clean_crime_data

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
from crime_data_library import calculate_missing_data

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
from crime_data_library import compute_crime_rate_by_year

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
from crime_data_library import top_crime_types

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
from crime_data_library import find_high_crime_areas

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
from crime_data_library import generate_timestamped_filename

filename = generate_timestamped_filename("cleaned_crime_data")
clean_df.to_csv(filename, index=False)
print(f"File saved as: {filename}")
```

Example output:

```
File saved as: cleaned_crime_data_2025-10-12_21-04-10.csv
```


# Function library overview and organization
The library consists of 21 total functions, at time of writing. An important detail to take note of regarding all functions in the library is that they are each designed to function with complete independence - that is, not relying on other functions in the library to be able to run. In terms of organization, the .py file for the function library uses some comment lines to break apart the functions into 5 core sections, which each represent the overarching purpose in the project that one group of functions, each achieving a more niche, specific goal, is intended to fulfill. These groups are.....
1. Data Ingestion
2. Data Cleaning
3. Data Transformation
4. Data Analysis
5. Data Storage & Utilities

# Contribution guidelines for team members
1. Ensure you have the latest code
2. Make your changes in the appropriate module
4. Write tests for new functionality
5. Update documentation as needed
6. Follow PEP 8 documentation guidelines
