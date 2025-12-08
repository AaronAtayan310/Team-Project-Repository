#!/usr/bin/env python3
"""
Crime Research Data Pipeline - Core Classes Demo Script

This script demonstrates the core classes in our crime research data 
pipeline project.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: OOP Class Implementation (Project 2)
"""

import tempfile
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys
import os

# Add src directory to path so we can import our library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from proj2_data_analysis_cls import *
from proj2_data_cleaning_cls import *
from proj2_data_ingestion_cls import *
from proj2_data_transformation_cls import *
from proj2_data_utilities_cls import *


def ingestion_class_demo():
    """Demonstrate the basic capabilities of the dataIngestion class such as printing objects via _str__ or __repr__, loading & validating csv files, fetching API data, etc."""
    print("❗ DATA INGESTION CLASS DEMO")
    print("=" * 50)
    
    # Create an instance
    ingestion = dataIngestion()
    print(ingestion)
    print(repr(ingestion))
    
    # Validate a CSV path
    is_valid = dataIngestion.validate_csv_path("example.csv")
    print(f"CSV path valid: {is_valid}")
    
    # Load CSV (example - will fail if file doesn't exist)
    # df = ingestion.load_csv("[input .csv file]")
    
    # Fetch API data (example)
    data = ingestion.fetch_api_data("https://api.nationalize.io/?name=nathaniel")
    
    # Check loaded sources
    print(f"Data sources: {ingestion._data_sources}")

    # Demonstrate different initialization options
    ingestion_custom = dataIngestion(default_timeout=5, track_sources=False)
    print("\nCustom initialized instance:")
    print(ingestion_custom)
    print(f"Custom timeout: {ingestion_custom.default_timeout}")
    print(f"Custom tracking: {ingestion_custom.track_sources}")
    
    # Demonstrate property setters
    ingestion.default_timeout = 15
    print(f"\nUpdated default timeout: {ingestion.default_timeout}")
    
    ingestion.track_sources = True
    print(f"Tracking enabled: {ingestion.track_sources}")
    
    # Show source tracking after API calls
    print(f"\nTracked sources after API calls: {len(ingestion._data_sources)}")
    if ingestion._data_sources:
        print("Latest source:", ingestion._data_sources[-1])
    
    # Demonstrate source clearing
    ingestion.clear_sources()
    print(f"\nSources after clearing: {len(ingestion._data_sources)}")


def cleaning_class_demo():
    """Demonstrate core dataCleaner class capabilities like initialization, handling missing values, text normalization, and object string representations."""
    print("\n\n❗ DATA CLEANING CLASS DEMO")
    print("=" * 50)
    
    # Create a small DataFrame with missing values and messy text
    df = pd.DataFrame(
        {
            "crime_type": ["Theft ", "ASSAULT", None, " burglary"],
            "value": [10, None, 30, 40],
            "count": [1, 2, None, 4],
        }
    )
    print("Original DataFrame:")
    print(df)

    # Initialize cleaner
    cleaner = dataCleaner(df, verbose=True)
    print("\nCleaner __repr__ output:")
    print(repr(cleaner))

    # Handle missing values using mean strategy on numeric columns
    cleaner.handle_missing_values(strategy="mean", columns=["value", "count"])
    print("\nDataFrame after handling missing values (mean strategy on numeric columns):")
    print(cleaner.df)

    # Normalize text column
    cleaner.normalize_text_column("crime_type", remove_special_chars=True)
    print("\nDataFrame after normalizing 'crime_type' column:")
    print(cleaner.df)

    # Show string summary
    print("\nCleaner __str__ summary:")
    print(str(cleaner))


def transformation_class_demo():
    """Demonstrate key dataTransformation class operations such as feature scaling, feature generation, and object representations on a small sample DataFrame."""
    print("\n\n❗ DATA TRANSFORMATION CLASS DEMO")
    print("=" * 50)
    
    # Create sample DataFrame
    df = pd.DataFrame(
        {
            "value": [100, 200, 300, 400],
            "count": [1, 2, 3, 4],
            "crime_type": ["theft", "assault", "theft", "burglary"],
        }
    )
    print("Original DataFrame for transformation:")
    print(df)

    # Initialize transformer
    transformer = dataTransformation(df)
    print("\nTransformer __repr__ output before transformations:")
    print(repr(transformer))

    # Scale numeric features
    transformer.scale_features(["value", "count"])
    print("\nDataFrame after scaling 'value' and 'count':")
    print(transformer.frame)

    # Generate derived feature
    transformer.generate_features()
    print("\nDataFrame after generating 'value_per_count' feature:")
    print(transformer.frame)

    # Show string representation (prints internal state)
    print("\nTransformer __str__ output:")
    print(str(transformer))


def analysis_class_demo():
    """Showcase key dataAnalysis class features including regression fitting, model evaluation, missing data calculation, and crime-specific aggregations."""
    print("\n\n❗ DATA ANALYSIS CLASS DEMO")
    print("=" * 50)
    
    # Create a simple DataFrame suitable for analysis
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-05-10", "2021-03-15", "2021-08-20"],
            "crime_type": ["theft", "assault", "theft", "burglary"],
            "population": [100000, 100000, 120000, 120000],
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [10.0, 20.0, 30.0, 40.0],
            "neighborhood": ["A", "A", "B", "B"],
        }
    )
    y = pd.Series([5.0, 10.0, 15.0, 20.0])

    # Initialize analysis object
    analysis = dataAnalysis(df)
    print("Analysis __repr__ output:")
    print(repr(analysis))

    # Run a simple regression
    model = analysis.run_regression(y)
    print("\nTrained LinearRegression model:")
    print(model)

    # Evaluate model using same targets as a simple demo
    metrics = analysis.evaluate_model(model, y_test=y)
    print("\nModel evaluation metrics:")
    print(metrics)

    # Calculate missing data percentages
    missing_pct = analysis.calculate_missing_data()
    print("\nMissing data percentage per column:")
    print(missing_pct)

    # Compute crime rate by year
    crime_rate_by_year = analysis.compute_crime_rate_by_year(population_col="population")
    print("\nCrime rate by year:")
    print(crime_rate_by_year)

    # Top crime types
    top_crimes = analysis.top_crime_types(n=2)
    print("\nTop 2 crime types:")
    print(top_crimes)

    # High crime areas
    high_crime_areas = analysis.find_high_crime_areas(area_col="neighborhood")
    print("\nHigh crime areas:")
    print(high_crime_areas)

    # Use __str__ (prints internal data and description)
    print("\nAnalysis __str__ output:")
    print(str(analysis))


def utilities_class_demo():
    """Demonstrate dataStorageUtils class functionalities for CSV/JSON IO, model serialization, hashing, and manifest creation by using some temporary files."""
    print("\n\n❗ DATA UTILITIES CLASS DEMO")
    print("=" * 50)
    
    # Use a temporary directory so our demo does not cause clutter
    with tempfile.TemporaryDirectory() as tmpdir:
        utils = dataStorageUtils(base_output_dir=tmpdir)
        print("Utilities __repr__ output:")
        print(repr(utils))

        # Create a small DataFrame and save/load as CSV
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_path = utils.save_to_csv(df, os.path.join(tmpdir, "demo.csv"), use_timestamp=False)
        print(f"\nCSV saved to: {csv_path}")
        loaded_df = utils.load_from_csv(str(csv_path))
        print("Loaded CSV DataFrame:")
        print(loaded_df)

        # Serialize and deserialize a simple model
        model = LinearRegression()
        model_path = utils.serialize_model(model, os.path.join(tmpdir, "model.pkl"))
        print(f"\nModel serialized to: {model_path}")
        loaded_model = utils.deserialize_model(str(model_path))
        print("Deserialized model:")
        print(loaded_model)

        # Save and load JSON
        json_data = {"step": "demo", "status": "success"}
        json_path = utils.save_to_json(json_data, os.path.join(tmpdir, "demo.json"), use_timestamp=False)
        print(f"\nJSON saved to: {json_path}")
        loaded_json = utils.load_from_json(str(json_path))
        print("Loaded JSON data:")
        print(loaded_json)

        # Compute a file hash for the CSV
        file_hash = utils.compute_file_hash(str(csv_path))
        print(f"\nComputed file hash for CSV: {file_hash}")

        # Create a simple pipeline manifest
        manifest_path = utils.create_pipeline_manifest(
            {"steps": ["ingestion", "cleaning", "analysis"], "status": "completed"},
            filepath=os.path.join(tmpdir, "manifest.json"),
        )
        print(f"\nPipeline manifest saved to: {manifest_path}")

        # Get directory size
        dir_size = utils.get_directory_size(tmpdir)
        print(f"\nTemporary directory size (bytes): {dir_size}")


def main():
    """Run all core class demonstration functions."""
    print("CRIME RESEARCH DATA PIPELINE - CORE CLASSES DEMO")
    print("=" * 60)
    print("This demo showcases how the core classes have adapted the function library to support crime analysis using an object-oriented approach, based on some small sample data.")

    ingestion_class_demo()
    cleaning_class_demo()
    transformation_class_demo()
    analysis_class_demo()
    utilities_class_demo()

    print("\n" + "=" * 60)
    print("All core class demonstrations have been ran successfully.")
    print("These core classes have successfully adapted the foundational function library using OOP principles to support more advanced project expansion over the remaining semester.")
    print("=" * 60)


if __name__ == "__main__":
    main()
