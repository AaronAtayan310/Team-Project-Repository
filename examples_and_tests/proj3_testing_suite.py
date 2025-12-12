"""
Crime Research Data Pipeline - Refactored & Advanced Classes Comprehensive Test Suite

This script tests inheritance, polymorphism, abstract base classes, and composition for
the advanced OOP implementation of the Crime Research Data Pipeline.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import json
import logging
from unittest.mock import MagicMock, patch, mock_open
from abc import ABC
from datetime import datetime

# Add src directory to path so we can import the relevant files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from proj3_base_classes import AbstractDataProcessor, AbstractDataSource
from proj3_data_ingestion_new import NewDataIngestion
from proj3_data_pipeline import DataPipeline
from proj3_data_processors import NewDataAnalysis, NewDataCleaner, NewDataTransformation
from proj3_data_sources import APIDataSource, CSVDataSource, DatabaseDataSource
from proj3_data_utilities_new import NewDataStorageUtils


# Define helper function for generating mock data so we can execute all tests
def create_mock_dataframe(rows=10, cols=3, missing_ratio=0.1, has_duplicates=False):
    """Creates a mock DataFrame for tests pertaining to the subclasses of AbstractDataProcessor."""
    data = {
        'numeric_col': np.random.rand(rows),
        'text_col': [f'Area {i}' for i in range(rows)],
        'date': pd.to_datetime(['2023-01-01'] * (rows//2) + ['2024-01-01'] * (rows - rows//2)),
        'value': np.arange(rows),
        'count': np.arange(rows) + 1,
        'population': np.full(rows, 100000),
        'crime_type': ['Theft', 'Assault'] * (rows//2),
        'neighborhood': [f'Hood {i}' for i in range(rows)]
    }
    
    df = pd.DataFrame(data)

    # Introduce missing values and introduce duplicates, return with simplicity to keep tests basic
    if missing_ratio > 0:
        mask = np.random.choice([False, True], size=df.shape, p=[1 - missing_ratio, missing_ratio])
        df[mask] = np.nan
    if has_duplicates and rows >= 2:
        df.iloc[0] = df.iloc[1]
    return df.drop(columns=['date', 'population', 'crime_type', 'neighborhood', 'text_col'], errors='ignore')


class TestAbstractBaseClasses(unittest.TestCase):
    """Test that abstract base classes enforce implementation."""

    def test_abstract_data_processor_cannot_be_instantiated(self):
        """Verify AbstractDataProcessor cannot be instantiated directly."""
        with self.assertRaises(TypeError) as context:
            AbstractDataProcessor(frame=pd.DataFrame())
        self.assertIn("Can't instantiate abstract class AbstractDataProcessor", str(context.exception))
    
    def test_abstract_data_source_cannot_be_instantiated(self):
        """Verify AbstractDataSource cannot be instantiated directly."""
        with self.assertRaises(TypeError) as context:
            AbstractDataSource(source_path="mock/path")
        self.assertIn("Can't instantiate abstract class AbstractDataSource", str(context.exception))

    def test_concrete_processor_enforces_abstract_methods(self):
        """Verify a mock class inheriting AbstractDataProcessor must implement methods."""
        # Define a mock class that forgets to implement 'validate'
        class IncompleteProcessor(AbstractDataProcessor):
            def process(self):
                return self

        # Attempting to instantiate the incomplete class should raise an error
        with self.assertRaises(TypeError) as context:
            IncompleteProcessor(frame=pd.DataFrame())
        
        self.assertIn("Can't instantiate abstract class IncompleteProcessor with abstract method 'validate'", str(context.exception))


class TestInheritance(unittest.TestCase):
    """Test inheritance hierarchies."""

    def test_data_processors_inherit_from_abstract_data_processor(self):
        """Verify concrete processor classes inherit from the correct base class."""
        processors = [NewDataAnalysis, NewDataCleaner, NewDataTransformation]
        for Processor in processors:
            with self.subTest(Processor=Processor.__name__):
                self.assertTrue(issubclass(Processor, AbstractDataProcessor),
                                f"{Processor.__name__} does not inherit from AbstractDataProcessor")
                self.assertTrue(issubclass(Processor, ABC), # Inheriting from ABC is required for ABCs
                                f"{Processor.__name__} is not an Abstract Base Class (ABC) subtype")
    
    def test_data_sources_inherit_from_abstract_data_source(self):
        """Verify concrete source classes inherit from the correct base class."""
        sources = [APIDataSource, CSVDataSource, DatabaseDataSource]
        for Source in sources:
            with self.subTest(Source=Source.__name__):
                self.assertTrue(issubclass(Source, AbstractDataSource),
                                f"{Source.__name__} does not inherit from AbstractDataSource")
                self.assertTrue(issubclass(Source, ABC),
                                f"{Source.__name__} is not an Abstract Base Class (ABC) subtype")


class TestPolymorphism(unittest.TestCase):
    """Test polymorphic behavior across different object types."""
    
    def setUp(self):
        """Set up useful mock data for testing."""
        self.mock_df = create_mock_dataframe(rows=5)

    # Quick patching for test maintenance    
    @patch('proj3_data_sources.APIDataSource.load')
    @patch('proj3_data_sources.APIDataSource.validate_source', return_value=True)
    
    def test_ingestion_polymorphism_with_api_source(self, mock_validate, mock_load):
        """Test NewDataIngestion handles APIDataSource polymorphically."""
        mock_load.return_value = self.mock_df.copy()
        ingestion = NewDataIngestion(track_sources=True)
        api_source = APIDataSource(source_path="mock_api")
        
        result_df = ingestion.load_from_source(api_source)
        
        mock_validate.assert_called_once()
        mock_load.assert_called_once()
        self.assertTrue(result_df.equals(self.mock_df))
        self.assertEqual(ingestion._data_sources[0]['source_type'], 'APIDataSource')
        
    # Quick patching for test maintenance
    @patch('proj3_data_sources.CSVDataSource.load')
    @patch('proj3_data_sources.CSVDataSource.validate_source', return_value=True)
    
    def test_ingestion_polymorphism_with_csv_source(self, mock_validate, mock_load):
        """Test NewDataIngestion handles CSVDataSource polymorphically."""
        mock_load.return_value = self.mock_df.copy()
        ingestion = NewDataIngestion(track_sources=True)
        csv_source = CSVDataSource(source_path="mock.csv")
        
        result_df = ingestion.load_from_source(csv_source)
        
        mock_validate.assert_called_once()
        mock_load.assert_called_once()
        self.assertTrue(result_df.equals(self.mock_df))
        self.assertEqual(ingestion._data_sources[0]['source_type'], 'CSVDataSource')
        
    def test_processor_polymorphism_process_and_validate(self):
        """Test that calling process() and validate() on different processors works generically."""        
        # Create a mock DataFrame that will pass all validations
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        processors = [
            NewDataCleaner(df.copy(), verbose=False),
            NewDataTransformation(df.copy(), verbose=False),
            NewDataAnalysis(df.copy(), verbose=False)
        ]

        for processor in processors:
            with self.subTest(Processor=processor.__class__.__name__):
                # Test polymorphic process() call
                self.assertIsInstance(processor.process(), AbstractDataProcessor, f"{processor.__class__.__name__}.process() did not return self/processor object.")
                
                # Test polymorphic validate() call (should pass for simple clean data)
                self.assertTrue(processor.validate(), f"{processor.__class__.__name__}.validate() failed unexpectedly.")


class TestComposition(unittest.TestCase):
    """Test composition relationships."""

    def setUp(self):
        """Set up useful mock data for testing."""
        self.mock_df = create_mock_dataframe(rows=5)
        
    # Quick patching for test maintenance
    @patch('proj3_data_sources.CSVDataSource.load')
    @patch('proj3_data_sources.CSVDataSource.validate_source', return_value=True)
    
    def test_ingestion_load_csv_uses_csv_data_source(self, mock_validate, mock_load):
        """Test NewDataIngestion composes and uses CSVDataSource for convenience method."""
        mock_load.return_value = self.mock_df
        ingestion = NewDataIngestion()
        mock_filepath = "test_data.csv"
        
        ingestion.load_csv(mock_filepath)
        
        # Verify that load_csv created a CSVDataSource and delegated to it
        self.assertTrue(mock_validate.called)
        self.assertTrue(mock_load.called)
        
    def test_data_pipeline_composition_initialization(self):
        """Test DataPipeline correctly holds instances of other components (HAS-A relationship)."""
        # Test creation with default components
        pipeline = DataPipeline(verbose=False)
        self.assertIsInstance(pipeline.ingestion, NewDataIngestion)
        self.assertIsInstance(pipeline.storage, NewDataStorageUtils)
        self.assertIsNone(pipeline.cleaner)
        
        # Test creation with injected components (dependency injection)
        mock_ingestion = MagicMock(spec=NewDataIngestion)
        mock_storage = MagicMock(spec=NewDataStorageUtils)
        
        pipeline_injected = DataPipeline(ingestion=mock_ingestion, storage=mock_storage, verbose=False)
        self.assertIs(pipeline_injected.ingestion, mock_ingestion)
        self.assertIs(pipeline_injected.storage, mock_storage)
        
    # Quick patching for test maintenance
    @patch('proj3_data_processors.NewDataCleaner.__init__', return_value=None)
    @patch('proj3_data_processors.NewDataCleaner.handle_missing_values')
    @patch('proj3_data_processors.NewDataCleaner', autospec=True)
    
    def test_data_pipeline_clean_composes_data_cleaner(self, MockCleanerClass, mock_handle_missing_values, mock_init):
        """Test DataPipeline instantiates and uses NewDataCleaner in the clean stage."""
        # Setup pipeline state
        pipeline = DataPipeline(verbose=False)
        pipeline.data = self.mock_df.copy()
        
        # Mock the instance created by the pipeline's call
        mock_cleaner_instance = MockCleanerClass.return_value
        mock_cleaner_instance.frame = self.mock_df.copy() # The data frame state after processing

        # Run the clean stage
        pipeline.clean(strategy='median')
        
        # 1. Test Composition (instantiation)
        MockCleanerClass.assert_called_once_with(self.mock_df, verbose=False)
        self.assertIs(pipeline.cleaner, mock_cleaner_instance)
        
        # 2. Test Delegation (method call)
        mock_cleaner_instance.handle_missing_values.assert_called_once_with(strategy='median')
        
        # 3. Test State Update
        self.assertTrue(pipeline.data.equals(mock_cleaner_instance.frame))


class TestSystemIntegration(unittest.TestCase):
    """Test complete system working together."""

    def setUp(self):
    """Set up simple mock data for end-to-end testing."""
        self.raw_df = pd.DataFrame({
            'numeric_col': [10.0, 20.0, np.nan],
            'count': [1, 2, 3],
            'value': [10, 20, 30]
        })
        self.mock_output_path = "output/processed_results.csv"

    # Quick patching for test maintenance
    @patch('proj3_data_utilities_new.NewDataStorageUtils.save_to_csv')
    @patch('proj3_data_ingestion_new.NewDataIngestion.load_csv')
    @patch('proj3_data_sources.CSVDataSource.validate_source', return_value=True)

    def test_run_full_pipeline_success(self, mock_validate, mock_load_csv, mock_save_to_csv):
        """Test the run_full_pipeline method orchestrates all stages correctly."""
        # Setup Mocks and initialize pipeline
        mock_load_csv.return_value = self.raw_df.copy()
        mock_save_to_csv.return_value = self.mock_output_path # Mock file path return
        pipeline = DataPipeline(verbose=False)
        
        # Run the full pipeline
        processed_df = pipeline.run_full_pipeline(
            source="mock_data.csv",
            output_path=self.mock_output_path,
            clean_strategy='mean',
            scale_columns=['numeric_col']
        )
        
        # Check ingestion is working in the integreated system (delegation)
        mock_load_csv.assert_called_once_with("mock_data.csv")
        self.assertIsNotNone(pipeline.data)
        
        # Check cleaning is working in the integreated system (processors and method chaining)
        self.assertIsInstance(pipeline.cleaner, NewDataCleaner)
        self.assertAlmostEqual(pipeline.data.iloc[2]['numeric_col'], 15.0) # Verifies that missing value was inputed (mean of 10.0 and 20.0 is 15.0)
        
        # Check transformation is working in the integreated system
        self.assertIsInstance(pipeline.transformer, NewDataTransformation)
        self.assertIn('value_per_count', pipeline.data.columns) # Verifies that generate_features was called (creates 'value_per_count')
        
        # Check analysis is working in the integreated system
        self.assertIsInstance(pipeline.analyzer, NewDataAnalysis)
        self.assertEqual(len(pipeline.analyzer.processing_history), 2) # Init + process()
        
        # Check storage is working in the integreated system (more delegation)
        mock_save_to_csv.assert_called_once()
        
        # Check pipeline history and summary checking is working in the integreated system
        self.assertTrue(len(pipeline.pipeline_history) >= 5, "Pipeline history must track all major steps.")
        summary = pipeline.get_summary()
        self.assertEqual(summary['data_shape'], (3, 4)) # 3 rows, 4 cols (3 original + 1 generated)
        self.assertEqual(summary['cleaning_operations'], 2) # Init + handle_missing_values()

        # Final return Check
        self.assertTrue(isinstance(processed_df, pd.DataFrame))


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL) # keep logging quieted for clean test output
    unittest.main()
