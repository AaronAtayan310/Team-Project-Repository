"""
Crime Research Data Pipeline - Refactored & Advanced Classes Demo Script

This script demonstrates the advanced OOP implementation of our crime
research data pipeline project satisfying all Project 3 requirements:
- Inheritance hierarchies
- Polymorphic behavior
- Abstract base classes
- Composition relationships

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

import pandas as pd
import numpy as np
import tempfile
from typing import List, Union
import sys
import os

# Add src directory to path so we can import the relevant code files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from proj3_base_classes import AbstractDataProcessor, AbstractDataSource
from proj3_data_ingestion_new import NewDataIngestion
from proj3_data_pipeline import DataPipeline
from proj3_data_processors import NewDataAnalysis, NewDataCleaner, NewDataIngestion
from proj3_data_sources import APIDataSource, CSVDataSource, DatabaseDataSource
from proj3_data_utilities_new import NewDataStorageUtils


# Define simulated data sources and helper function for crafting mock data so we can run all demonstrations
TEMP_CSV_PATH = os.path.join(tempfile.gettempdir(), "crime_data_mock.csv")
OUTPUT_PATH = os.path.join(tempfile.gettempdir(), "processed_crime_results.csv")
def create_mock_data_file():
    """Generates a simple CSV file with missing values for later demo functions."""
    print(f"-> Creating mock data file at: {TEMP_CSV_PATH}")
    data = {
        'Incident_ID': range(1, 11),
        'Year': [2020, 2021] * 5,
        'Crime_Rate': [250, 310, np.nan, 280, 260, 320, 270, 290, 300, 305],
        'Num_Incidents': [100, 120, 90, 110, 105, np.nan, 95, 115, 125, 130],
        'Population': [40000, 40000, 35000, 45000, 42000, 40000, 38000, 41000, 43000, 44000],
        'Area': ['North', 'South', 'North', 'East', 'West', 'South', 'East', 'North', 'West', 'East']
    }
    df = pd.DataFrame(data)
    df.to_csv(TEMP_CSV_PATH, index=False)
    print("-> Mock data created successfully.")
    return TEMP_CSV_PATH
    

def demonstrate_inheritance():
    """Demonstrate inheritance hierarchy."""
    print("=" * 60)
    print("INHERITANCE DEMONSTRATION")
    print("=" * 60)

    # Show that data sources are inheriting from AbstractDataSource
    csv_source = CSVDataSource(TEMP_CSV_PATH)
    api_source = APIDataSource("https://mock.api/crime")

    print("\nInheritance: Observe data sources inheriting from AbstractDataSource")
    sources = [csv_source, api_source]
    for source in sources:
        base_class = source.__class__.__bases__[0].__name__
        is_sub = issubclass(type(source), AbstractDataSource)
        print(f"  - {type(source).__name__:20s} inherits from {base_class:20s}: {is_sub}")

    # Show that data processors are inheriting from AbstractDataProcessor
    mock_frame = pd.DataFrame()
    cleaner = NewDataCleaner(mock_frame, verbose=False)
    transformer = NewDataTransformation(mock_frame, verbose=False)

    print("\nData Processors inheriting from AbstractDataProcessor:")
    processors = [cleaner, transformer]
    for processor in processors:
        base_class = processor.__class__.__bases__[0].__name__
        is_sub = issubclass(type(processor), AbstractDataProcessor)
        print(f"  - {type(processor).__name__:20s} inherits from {base_class:20s}: {is_sub}")

def demonstrate_polymorphism():
    """Demonstrate polymorphic behavior."""
    print("\n" + "=" * 60)
    print("POLYMORPHISM DEMONSTRATION")
    print("=" * 60)

    # Firstly showing polymorphism in data ingestion (load_data on AbstractDataSource) 
    print("\nPolymorphism: Observe polymorphic data loading (the ingestion component)")

    ingestion_manager = NewDataIngestion(track_sources=True)

    # Create instances of different concrete AbstractDataSource subtypes
    sources: List[Union[CSVDataSource, APIDataSource]] = [
        CSVDataSource(TEMP_CSV_PATH),
        APIDataSource("https://api.crime.gov/data")
    ]

    for source in sources:
        # 'source.load()' is the same method being called repeatedly and the NewDataIngestion manager calls the generic 'load' method on the source object (mock load on APIDataSource to prevent real api calling)
        if isinstance(source, APIDataSource):
            df_loaded = pd.DataFrame({"msg": [f"Mock data from {source.__class__.__name__}"]})
        else:
            df_loaded = ingestion_manager.load_from_source(source, source.__class__.__name__)
        
        print(f"  - Source Type: {source.__class__.__name__:18s} -> Data Shape: {df_loaded.shape}")

    # Secondly showing polymorphism in data processing (process() on AbstractDataProcessor) 
    print("\nPolymorphism Furthered: Observe polymorphic data processing (the processor components)")

    df = pd.DataFrame({'colA': [1, 2], 'colB': [3, 4]})
    processors: List[AbstractDataProcessor] = [
        NewDataCleaner(df.copy()),
        NewDataTransformation(df.copy()),
        NewDataAnalysis(df.copy())
    ]

    for processor in processors:
        # 'processor.process()' is the same method being called repeatedly and the implementation of 'process()' varies for each concrete subclass
        processed_obj = processor.process()
        print(f"  - Processor Type: {processed_obj.__class__.__name__:20s} -> Executed specialized process()")
        # Demonstrate the method chaining return (must return the object instance)
        print(f"  - Return Value Type: {processed_obj.__class__.__name__} (Proves consistent interface)")


def demonstrate_abstract_base_classes():
    """Demonstrate abstract base class usage."""
    print("\n" + "=" * 60)
    print("ABSTRACT BASE CLASS DEMONSTRATION")
    print("=" * 60)

    print("\nAbstract Base Classes / ABCs: Observe that there is an enforced contract/interface for all concrete subclasses.")

    print("\nThe AbstractDataSource contract...")
    print("  - Requires concrete implementation of `load()` (how to read the data).")
    print("  - Requires concrete implementation of `validate_source()` (checks path/connection).")
    print("\n  -> Example: CSVDataSource implements `load` using pandas.read_csv.")
    
    print("\nThe AbstractDataProcessor contract...")
    print("  - Requires concrete implementation of `process()` (the main manipulation logic).")
    print("  - Requires concrete implementation of `validate()` (checks data integrity/quality).")
    print("\n  -> Example: NewDataCleaner implements `process` by running missing value/duplicate handling.")
    
    # Show that failure occurs if instantiation is attempted
    print("\nDemonstrating enforcement (cannot instantiate AbstractDataSource):")
    try:
        AbstractDataSource(source_path="error")
    except TypeError as e:
        print(f"  - SUCCESS: Caught expected error: {e}")
    except Exception as e:
        print(f"  - FAILURE: Caught unexpected error: {e}")


def demonstrate_composition():
    """Demonstrate composition relationships."""
    print("\n" + "=" * 60)
    print("COMPOSITION DEMONSTRATION")
    print("=" * 60)

    print("\nComposition: Observe that the DataPipeline is the central orchestrator and 'has-a' relationship with other components.")

    # See that DataPipeline HAS-A NewDataIngestion and NewDataStorageUtils (via dependency injection) - data pipelines have ingestion and storage responsibilities)
    ingestion = NewDataIngestion(track_sources=False)
    storage = NewDataStorageUtils(output_dir="/tmp/results")
    
    pipeline = DataPipeline(ingestion=ingestion, storage=storage, verbose=False)
    
    print("\nA DataPipeline HAS-A NewDataIngestion and NewDataStorageUtils (via dependency injection):")
    print(f"  - Pipeline contains NewDataIngestion object: {pipeline.ingestion.__class__.__name__}")
    print(f"  - Pipeline contains NewDataStorageUtils object: {pipeline.storage.__class__.__name__}")

    # See that DataPipeline HAS-A processor instances (created during runtime) and that the clean method will instantiate the cleaner
    pipeline.load_data(TEMP_CSV_PATH)
    pipeline.clean(strategy='median') 

    print("\nA DataPipeline HAS-A NewDataCleaner instance (created/contained during clean stage):")
    print(f"  - Pipeline contains Cleaner object: {pipeline.cleaner.__class__.__name__}")
    print(f"  - Cleaner reports {len(pipeline.cleaner.processing_history)} internal operation(s).")
    
    # See that NewDataIngestion HAS-A CSVDataSource (internal composition for convenience method)
    ingestion_manager = NewDataIngestion()
    print("\nA NewDataIngestion HAS-A CSVDataSource (used internally for load_csv convenience):")
    # This call internally creates and uses a CSVDataSource object
    ingestion_manager.load_csv(TEMP_CSV_PATH) 
    print("  - NewDataIngestion successfully delegated file loading to an internal CSVDataSource.")


def demonstrate_complete_system():
    """Demonstrate the complete integrated system."""
    print("\n" + "=" * 60)
    print("COMPLETE SYSTEM DEMONSTRATION")
    print("=" * 60)

    mock_path = create_mock_data_file()

    print("\nCompleteness: Observe the running of the full DataPipeline in a single sequence (method chaining):")
    
    # Instantiate the master pipeline object
    pipeline = DataPipeline(verbose=True)

    # Run the complete end-to-end pipeline using method chaining and the wrapper method
    processed_df = pipeline.run_full_pipeline(
        source=mock_path,
        output_path=OUTPUT_PATH,
        clean_strategy='mean', # Impute missing values with mean
        scale_columns=['Crime_Rate', 'Num_Incidents']
    )
    
    print("\n" + "-" * 60)
    print("Pipeline Execution Complete. Final State and Summary:")
    print("-" * 60)
    
    # Print pipeline summary using the overridden __str__ method
    print(pipeline)

    # Print structural summary
    summary = pipeline.get_summary()
    print("\nFormal Summary Check (get_summary()):")
    for key, value in summary.items():
        print(f"  {key:25s}: {value}")

    print(f"\nFinal processed data saved to: {OUTPUT_PATH}")
    print(f"Data Head:\n{processed_df.head(3)}")

    # Clean up the temp file
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        # os.remove(mock_path) # Leaving source file for potential inspection


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("CRIME RESEARCH DATA PIPELINE - PROJECT 3 DEMONSTRATION")
    print("Object-Oriented Programming with Inheritance & Polymorphism")
    print("=" * 60)

    # Create the source file once at the start
    mock_file_path = create_mock_data_file()
    
    demonstrate_inheritance()
    demonstrate_polymorphism()
    demonstrate_abstract_base_classes()
    demonstrate_composition()
    demonstrate_complete_system()

    # Clean up the mock source file created at the beginning
    if os.path.exists(mock_file_path):
        os.remove(mock_file_path)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThis system demonstrates:")
    print("  ✓ Inheritance hierarchies")
    print("  ✓ Polymorphic behavior (methods behave differently per type)")
    print("  ✓ Abstract base classes (enforce interface contracts)")
    print("  ✓ Composition relationships (has-a relationships)")
    print("  ✓ Complete system integration")
    print("\nAll Project 3 requirements satisfied!")


if __name__ == "__main__":
    main()
