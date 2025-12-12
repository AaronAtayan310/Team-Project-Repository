"""
Crime Research Data Pipeline - Function Library Init File

This module defines src as a python package for this repository, allowing
for key imports throughout all intermittent project implementations and
later demonstrations of key requirements for each implementation.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: All (Projects 1, 2, 3 & 4)
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "INST326 Crime Research Data Pipeline Project Team"
__email__ = "aatayan@terpmail.umd.edu"
__description__ = "Crime Research Data Pipeline Function Library for INST326"

# Import main functions from the crime research data pipeline library for easy access - Project 1 feature
from .proj1_crime_data_library import (
    # Data Ingestion Functions
    load_csv,
    fetch_api_data,
    validate_csv_path,
    
    # Data Cleaning Functions
    handle_missing_values,
    normalize_text_column,
    standardize_column_names,
    remove_outliers_iqr,
    clean_crime_data,
    
    # Data Transformation Functions
    scale_features,
    generate_features,
    
    # Data Analysis Functions
    compute_summary_stats,
    run_regression,
    evaluate_model,
    calculate_missing_data,
    compute_crime_rate_by_year,
    top_crime_types,
    find_high_crime_areas, 
    
    # Data Storage & Utility Functions
    save_to_csv,
    serialize_model,
    log_pipeline_step,
    generate_timestamped_filename
)

# Import core classes for easy access - Project 2 feature
from .proj2_data_analysis_cls import DataAnalysis
from .proj2_data_cleaning_cls import DataCleaner
from .proj2_data_ingestion_cls import DataIngestion
from .proj2_data_transformation_cls import DataTransformation
from .proj2_data_utilities_cls import DataStorageUtils

# Import refactored & advanced classes for easy access - Project 3 feature
from .proj3_base_classes import AbstractDataProcessor, AbstractDataSource
from .proj3_data_ingestion_new import NewDataIngestion
from .proj3_data_pipeline import DataPipeline
from .proj3_data_processors import NewDataAnalysis, NewDataCleaner, NewDataTransformation
from .proj3_data_sources import APIDataSource, CSVDataSource, DatabaseDataSource
from .proj3_data_utilities_new import NewDataStorageUtils

# Define what gets imported with "import *" statements - Project 1/2/3/4 feature
__all__ = [
    # Data Ingestion (Initial Work) - Project 1 feature
    'load_csv',
    'fetch_api_data',
    'validate_csv_path',
    
    # Data Cleaning (Initial Work) - Project 1 feature
    'handle_missing_values',
    'normalize_text_column',
    'standardize_column_names',
    'remove_outliers_iqr',
    'clean_crime_data',
    
    # Data Transformation (Initial Work) - Project 1 feature
    'scale_features',
    'generate_features',
    
    # Data Analysis (Initial Work) - Project 1 feature
    'compute_summary_stats',
    'run_regression',
    'evaluate_model',
    'calculate_missing_data',
    'compute_crime_rate_by_year',
    'top_crime_types',
    'find_high_crime_areas',
    
    # Data Storage & Utility (Initial Work) - Project 1 feature
    'save_to_csv',
    'serialize_model',
    'log_pipeline_step',
    'generate_timestamped_filename', 

    # Crime Research Data Pipeline Core Classes - Project 2 feature
    'DataAnalysis',
    'DataCleaner',
    'DataIngestion',
    'DataTransformation',
    'DataStorageUtils',

    # Crime Research Data Pipeline Refactored & Advanced Classes - Project 3 feature
    'AbstractDataProcessor',
    'AbstractDataSource',
    'NewDataIngestion',
    'DataPipeline',
    'NewDataAnalysis',
    'NewDataCleaner',
    'NewDataIngestion',
    'APIDataSource',
    'CSVDataSource',
    'DatabaseDataSource',
    'NewDataStorageUtils'
]

# Convenience groupings of functions in the library for easier access - Project 1 feature
DATA_INGESTION_FUNCTIONS = [
    'load_csv',
    'fetch_api_data',
    'validate_csv_path',
]

DATA_CLEANING_FUNCTIONS = [
    'handle_missing_values',
    'normalize_text_column',
    'standardize_column_names',
    'remove_outliers_iqr',
    'clean_crime_data',
]

DATA_TRANSFORMATION_FUNCTIONS = [
    'scale_features',
    'generate_features',
]

DATA_ANALYSIS_FUNCTIONS = [
    'compute_summary_stats',
    'run_regression',
    'evaluate_model',
    'calculate_missing_data',
    'compute_crime_rate_by_year',
    'top_crime_types',
    'find_high_crime_areas',
]

DATA_STORAGE_AND_UTILITIES_FUNCTIONS = [
    'save_to_csv',
    'serialize_model',
    'log_pipeline_step',
    'generate_timestamped_filename'
]

def get_function_categories(): # Project 1 feature
    """Get a dictionary of function categories and their functions.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping category names to function lists
        
    Example:
        >>> categories = get_function_categories()
        >>> print(categories['ingestion'])
        ['load_csv', 'fetch_api_data', ...]
    """
    return {
        'ingestion': DATA_INGESTION_FUNCTIONS,
        'cleaning': DATA_CLEANING_FUNCTIONS,
        'transformation': DATA_TRANSFORMATION_FUNCTIONS,
        'analysis': DATA_ANALYSIS_FUNCTIONS,
        'storage_and_utilities': DATA_STORAGE_AND_UTILITIES_FUNCTIONS
    }

def list_all_functions(): # Project 1 feature
    """List all available functions in the library.
    
    Returns:
        List[str]: Alphabetically sorted list of all function names
        
    Example:
        >>> functions = list_all_functions()
        >>> print(f"Total functions: {len(functions)}")
        Total functions: 21
    """
    lib_functions = __all__[:21]
    return sorted(lib_functions)

def get_library_info(): # Project 1 feature
    """Get library metadata and information.
    
    Returns:
        Dict[str, str]: Dictionary with library information
        
    Example:
        >>> info = get_library_info()
        >>> print(f"Version: {info['version']}")
        Version: 1.0.0
    """
    lib_functions = __all__[:21]
    return {
        'name': 'Crime Research Data Pipeline Function Library',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'total_functions': len(lib_functions),
        'categories': list(get_function_categories().keys()),
    }


