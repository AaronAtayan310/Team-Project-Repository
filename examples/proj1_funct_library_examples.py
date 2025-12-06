#!/usr/bin/env python3
"""
Crime Research Data Pipeline - Function Library Demo Script

This script demonstrates the key functions in our crime research data 
pipeline library with some practical scenarios.

Author: INST326 Crime Research Data Pipeline Project Team
"""

import sys
import os

# Add src directory to path so we can import our library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crime_data_library import *

def demo_data_ingestion():
    """Demonstrate data ingestion from different formats."""
    print("DATA INGESTION DEMO")
    print("=" * 50)

    # Demo: loading a CSV file (fake, for example purposes)
    filepath = "sample_crime_data.csv"
    try:
        data = load_csv(filepath)
        print(f"Loaded CSV with {len(data)} records from '{filepath}'")
    except FileNotFoundError:
        print(f"File '{filepath}' not found - skipping CSV loading demo.")
    
    # Demo: fetching API data (fake, for example purposes)
    test_url = "https://api.example.com/crime"
    try:
        api_data = fetch_api_data(test_url)
        print(f"API data fetched successfully with keys: {list(api_data.keys())}")
    except Exception as e:
        print(f"API data fetch failed: {e}")

    # Validate CSV path
    print(f"Is '{filepath}' a valid CSV path? {validate_csv_path(filepath)}")
    print()

def demo_data_cleaning():
    """Demonstrate cleaning capabilities for data / dataframes in the pipeline."""
    print("DATA CLEANING DEMO")
    print("=" * 50)

    # Example DataFrame for the demo
    import pandas as pd
    demo_df = pd.DataFrame({
        'Crime Type': ['Theft', 'Robbery', 'Assault', None, 'Theft'],
        'Value': [100, 200, None, 150, 100],
        'Region': ['A', 'B', 'A', 'B', 'A']
    })

    print("Original DataFrame:")
    print(demo_df)

    cleaned = handle_missing_values(demo_df, strategy='mean')
    print("\nAfter handling missing values:")
    print(cleaned)

    normalized = normalize_text_column(cleaned, 'Crime Type')
    print("\nAfter normalizing text:")
    print(normalized)

    standardized_cols = standardize_column_names(list(demo_df.columns))
    print("\nStandardized column names:")
    print(standardized_cols)

    print("\nRemoving outliers in 'Value' column (if present):")
    try:
        no_outliers = remove_outliers_iqr(normalized, 'Value')
        print(no_outliers)
    except Exception as e:
        print("Outlier removal failed:", e)

    print("\nGeneral cleaning for crime data:")
    print(clean_crime_data(demo_df))
    print()

def demo_data_transformation():
    """Demonstrate feature generation and scaling capabilities."""
    print("DATA TRANSFORMATION DEMO")
    print("=" * 50)

    import pandas as pd
    df = pd.DataFrame({
        'count': [10, 20, 30, 40, 50],
        'rate': [1.5, 2.0, 2.5, 3.0, 3.5]
    })

    print("Original data:")
    print(df)

    scaled_df = scale_features(df, ['count', 'rate'])
    print("\nScaled features (standardized):")
    print(scaled_df)

    transformed_df = generate_features(df)
    print("\nData with generated features:")
    print(transformed_df)
    print()

def demo_data_analysis():
    """Demonstrate core analysis capabilities of the library."""
    print("DATA ANALYSIS DEMO")
    print("=" * 50)

    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        'crime_type': ['theft', 'assault', 'theft', 'robbery', 'theft', 'burglary', 'robbery'],
        'value': [100, 230, 140, 200, 120, 320, 180],
        'date': pd.date_range("2024-01-01", periods=7, freq="M"),
        'population': [10000, 10000, 10000, 10000, 10000, 10000, 10000],
        'neighborhood': ['Downtown', 'East', 'Downtown', 'West', 'North', 'East', 'West']
    })

    print("Summary statistics:")
    print(compute_summary_stats(df))

    print("\nRunning simple regression model:")
    from sklearn.model_selection import train_test_split

    X = df[['value']]
    y = np.arange(len(df))  # Dummy target
    model = run_regression(X, y)
    print(f"Trained model coefficients: {model.coef_}")

    # Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trained_model = run_regression(X_train, y_train)
    results = evaluate_model(trained_model, X_test, y_test)
    print(f"Evaluation - MSE: {results['mse']}")

    print("\nMissing data:")
    print(calculate_missing_data(df))

    print("\nCrime rate by year calculation:")
    print(compute_crime_rate_by_year(df))

    print("\nTop crime types:")
    print(top_crime_types(df, n=3))

    print("\nHigh crime area detection:")
    print(find_high_crime_areas(df))
    print()

def demo_data_storage_utilities():
    """Demonstrate utility capabilities for storage and logging."""
    print("DATA STORAGE & UTILITIES DEMO")
    print("=" * 50)
    import pandas as pd
    import pickle

    # Demo DataFrame to save
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    filename = "test_output.csv"
    save_to_csv(df, filename)
    print(f"DataFrame saved to {filename}")

    # Serialize and reload a dummy model
    dummy_model = {'param': 1}
    model_file = "dummy_model.pkl"
    serialize_model(dummy_model, model_file)
    print(f"Serialized dummy model to {model_file}")
    print("Deserializing model to verify...")
    with open(model_file, 'rb') as f:
        reloaded = pickle.load(f)
    print("Reloaded model:", reloaded)

    # Log a step
    log_pipeline_step("DemoStep", "completed")
    print("Pipeline step logged.")

    # Generate a timestamped filename
    timestamped_name = generate_timestamped_filename("summary_report")
    print(f"Timestamped filename generated: {timestamped_name}")
    print()

def main():
    """Run all demonstration functions."""
    print("CRIME DATA FUNCTION LIBRARY - DEMO SCRIPT")
    print("=" * 60)
    print("This demo showcases how the function library supports crime analysis, based on some small sample data.")

    demo_data_ingestion()
    demo_data_cleaning()
    demo_data_transformation()
    demo_data_analysis()
    demo_data_storage_utilities()

    print("=" * 60)
    print("All demo's have been ran successfully.")
    print("This function library has built a solid foundation for object-oriented expansion on the project over the semester.")
    print("=" * 60)


if __name__ == "__main__":
    main()
