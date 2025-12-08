"""
Crime Research Data Pipeline - Function Library Init File

This module defines src as a python package for this repository, allowing
for some key imports throughout the project.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Function Library Development (Project 1)
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "INST326 Crime Research Data Pipeline Project Team"
__email__ = "aatayan@terpmail.umd.edu"
__description__ = "Crime Research Data Pipeline Function Library for INST326"
__license__ = "MIT"

# Import main functions for easy access
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
    find_high_crime_areas
    
    # Data Storage & Utility Functions
    save_to_csv,
    serialize_model,
    log_pipeline_step,
    generate_timestamped_filename
)

# Define what gets imported with "from proj1_crime_data_library import *"
__all__ = [
    # Data Ingestion
    'load_csv',
    'fetch_api_data',
    'validate_csv_path',
    
    # Data Cleaning
    'handle_missing_values',
    'normalize_text_column',
    'standardize_column_names',
    'remove_outliers_iqr',
    'clean_crime_data',
    
    # Data Transformation
    'scale_features',
    'generate_features',
    
    # Data Analysis
    'compute_summary_stats',
    'run_regression',
    'evaluate_model',
    'calculate_missing_data',
    'compute_crime_rate_by_year',
    'top_crime_types',
    'find_high_crime_areas',
    
    # Data Storage & Utility
    'save_to_csv',
    'serialize_model',
    'log_pipeline_step',
    'generate_timestamped_filename'
]

# Convenience function groupings for easier access
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

def get_function_categories():
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

def list_all_functions():
    """List all available functions in the library.
    
    Returns:
        List[str]: Alphabetically sorted list of all function names
        
    Example:
        >>> functions = list_all_functions()
        >>> print(f"Total functions: {len(functions)}")
        Total functions: 21
    """
    return sorted(__all__)

def get_library_info():
    """Get library metadata and information.
    
    Returns:
        Dict[str, str]: Dictionary with library information
        
    Example:
        >>> info = get_library_info()
        >>> print(f"Version: {info['version']}")
        Version: 1.0.0
    """
    return {
        'name': 'Crime Research Data Pipeline Function Library',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'total_functions': len(__all__),
        'categories': list(get_function_categories().keys()),
        'license': __license__
    }



