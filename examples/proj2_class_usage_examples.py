#!/usr/bin/env python3
"""
Crime Research Data Pipeline - Core Classes Demo Script

This script demonstrates the core classes in our crime research data 
pipeline project.

Author: INST326 Crime Research Data Pipeline Project Team
Course: INST326 - Object-Oriented Programming for Information Science
Project: OOP Class Implementation (Project 2)
"""

# ---------------------------------------------------------------------------
# 1. EXAMPLE USAGE : DATA INGESTION CLASS
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
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

# ---------------------------------------------------------------------------
# 2. EXAMPLE USAGE : DATA CLEANING CLASS
# ---------------------------------------------------------------------------
 
# INSERT CODE HERE

# ---------------------------------------------------------------------------
# 3. EXAMPLE USAGE : DATA TRANSFORMATION CLASS
# ---------------------------------------------------------------------------
 
# INSERT CODE HERE

# ---------------------------------------------------------------------------
# 4. EXAMPLE USAGE : DATA ANALYSIS CLASS
# ---------------------------------------------------------------------------
 
# INSERT CODE HERE

# ---------------------------------------------------------------------------
# 5. EXAMPLE USAGE : DATA STORAGE & UTILITIES CLASS
# ---------------------------------------------------------------------------
 
# INSERT CODE HERE
 
