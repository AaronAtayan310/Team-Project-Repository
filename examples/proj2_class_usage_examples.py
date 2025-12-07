#!/usr/bin/env python3
"""
Crime Research Data Pipeline - Core Classes Demo Script

This script demonstrates the core classes in our crime research data 
pipeline project.

Author: INST326 Crime Research Data Pipeline Project Team
Course: INST326 - Object-Oriented Programming for Information Science
Project: OOP Class Implementation (Project 2)
"""

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


def cleaning_class_demo():
    """Insert docstring here."""
    print("\n\n❗ DATA CLEANING CLASS DEMO")
    print("=" * 50)
    
    pass


def transformation_class_demo():
    """Insert docstring here."""
    print("\n\n❗ DATA TRANSFORMATION CLASS DEMO")
    print("=" * 50)
    
    pass


def analysis_class_demo():
    """Insert docstring here."""
    print("\n\n❗ DATA ANALYSIS CLASS DEMO")
    print("=" * 50)
    
    pass


def utilities_class_demo():
    """Insert docstring here."""
    print("\n\n❗ DATA UTILITIES CLASS DEMO")
    print("=" * 50)
    
    pass


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
