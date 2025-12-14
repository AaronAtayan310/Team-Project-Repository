"""
Crime Research Data Pipeline - Project 4 Testing Suite

Comprehensive test suite for all Project 4 enhanced classes and features.
Tests cover: inheritance, polymorphism, composition, registries, temporal
tracking, quality validation, and complete system integration.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os
from datetime import datetime

# Add src directory to path so we can import the relevant code files
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))    

from src.proj4_data_source import DataSource
from src.proj4_data_processor import DataProcessor
from src.proj4_specialized_sources import CSVCrimeDataSource, APICrimeDataSource, DatabaseCrimeDataSource
from src.proj4_specialized_processors import CrimeDataAnalysis, CrimeDataCleaner, CrimeDataTransformation
from src.proj4_data_quality_standards import DataQualityStandards, QualityLevel, ReportingStandard
from src.proj4_data_ingestion import CrimeDataIngestion
from src.proj4_data_utilities import CrimeDataStorageUtils
from src.proj4_pipeline_manager import PipelineManager



class TestEnhancedDataSource(unittest.TestCase):
    """Test enhanced DataSource base class features."""
    
    def setUp(self):
        """Set up test fixtures."""
        DataSource.clear_registry()
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_csv = self.test_dir / "test_data.csv"
        
        # Create test CSV
        df = pd.DataFrame({
            'incident_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'crime_type': ['Theft', 'Assault', 'Burglary'],
            'location': ['Downtown', 'Eastside', 'Westside']
        })
        df.to_csv(self.test_csv, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        DataSource.clear_registry()
    
    def test_uuid_identification(self):
        """Test that sources get UUID identification."""
        source = CSVCrimeDataSource(str(self.test_csv))
        
        self.assertIsNotNone(source.source_id)
        self.assertTrue(len(source.source_id) > 0)
        self.assertIsInstance(source.created_at, datetime)
    
    def test_registry_pattern(self):
        """Test that sources are automatically registered."""
        source1 = CSVCrimeDataSource(str(self.test_csv), name="source1")
        source2 = CSVCrimeDataSource(str(self.test_csv), name="source2")
        
        all_sources = DataSource.get_all_sources()
        self.assertEqual(len(all_sources), 2)
        
        # Test retrieval by ID
        retrieved = DataSource.get_source_by_id(source1.source_id)
        self.assertEqual(retrieved.source_id, source1.source_id)
    
    def test_temporal_tracking(self):
        """Test temporal tracking of load operations."""
        source = CSVCrimeDataSource(str(self.test_csv))
        
        # Load data
        df = source.load()
        
        # Check temporal tracking
        self.assertIsNotNone(source.last_loaded_at)
        self.assertEqual(source.load_count, 1)
        self.assertEqual(len(source.load_history), 1)
        
        # Load again
        df = source.load()
        self.assertEqual(source.load_count, 2)
        self.assertEqual(len(source.load_history), 2)
    
    def test_load_statistics(self):
        """Test load statistics calculation."""
        source = CSVCrimeDataSource(str(self.test_csv))
        source.load()
        source.load()
        
        stats = source.get_load_statistics()
        
        self.assertEqual(stats['total_loads'], 2)
        self.assertEqual(stats['successful_loads'], 2)
        self.assertEqual(stats['failed_loads'], 0)
        self.assertEqual(stats['success_rate'], 1.0)
        self.assertGreater(stats['total_rows_loaded'], 0)
    
    def test_data_lineage(self):
        """Test data lineage tracking."""
        source = CSVCrimeDataSource(str(self.test_csv))
        df = source.load()
        
        lineage = source.get_data_lineage()
        
        self.assertIn('source_id', lineage)
        self.assertIn('source_type', lineage)
        self.assertIn('last_data_checksum', lineage)
        self.assertIn('last_data_shape', lineage)
        self.assertEqual(lineage['source_type'], 'CSVCrimeDataSource')


class TestEnhancedDataProcessor(unittest.TestCase):
    """Test enhanced DataProcessor base class features."""
    
    def setUp(self):
        """Set up test fixtures."""
        DataProcessor.clear_registry()
        self.df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'count': [10, 20, 30, 40, 50]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        DataProcessor.clear_registry()
    
    def test_uuid_identification(self):
        """Test that processors get UUID identification."""
        processor = CrimeDataAnalysis(self.df)
        
        self.assertIsNotNone(processor.processor_id)
        self.assertTrue(len(processor.processor_id) > 0)
        self.assertIsInstance(processor.created_at, datetime)
    
    def test_registry_pattern(self):
        """Test that processors are automatically registered."""
        proc1 = CrimeDataAnalysis(self.df, name="analyzer1")
        proc2 = CrimeDataCleaner(self.df, name="cleaner1")
        
        all_processors = DataProcessor.get_all_processors()
        self.assertEqual(len(all_processors), 2)
        
        # Test retrieval by type
        analyzers = DataProcessor.get_processors_by_type('CrimeDataAnalysis')
        self.assertEqual(len(analyzers), 1)
    
    def test_performance_monitoring(self):
        """Test performance metrics collection."""
        processor = CrimeDataAnalysis(self.df)
        processor.process()
        
        metrics = processor.get_performance_metrics()
        
        self.assertIn('total_operations', metrics)
        self.assertIn('total_time_seconds', metrics)
        self.assertIn('average_operation_time', metrics)
        self.assertGreater(metrics['total_operations'], 0)
    
    def test_data_quality_tracking(self):
        """Test data quality metrics."""
        processor = CrimeDataAnalysis(self.df)
        
        quality = processor.get_data_quality_metrics()
        
        self.assertIn('completeness_score', quality)
        self.assertIn('total_rows', quality)
        self.assertIn('missing_cells', quality)
        self.assertEqual(quality['total_rows'], 5)
    
    def test_original_data_preservation(self):
        """Test that original data is preserved."""
        processor = CrimeDataCleaner(self.df)
        original_shape = processor.original_frame.shape
        
        # Modify data
        processor.handle_missing_values()
        
        # Original should be unchanged
        self.assertEqual(processor.original_frame.shape, original_shape)
        
        # Comparison should work
        comparison = processor.compare_to_original()
        self.assertIn('original_shape', comparison)


class TestDataQualityStandards(unittest.TestCase):
    """Test DataQualityStandards business logic class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            'incident_id': [1, 2, 3, 4, 5],
            'date': pd.date_range('2024-12-01', periods=5),
            'crime_type': ['Theft', 'Assault', None, 'Burglary', 'Robbery'],
            'location': ['A', 'B', 'C', 'D', 'E'],
            'jurisdiction': ['MD', 'MD', 'MD', 'MD', 'MD'],
            'reported_date': pd.date_range('2024-12-01', periods=5)
        })
        
        self.standards = DataQualityStandards("Maryland", "FBI_UCR")
    
    def test_class_level_thresholds(self):
        """Test class-level quality thresholds."""
        # Test threshold retrieval
        completeness_excellent = DataQualityStandards.get_threshold('completeness', 'excellent')
        self.assertEqual(completeness_excellent, 0.98)
        
        timeliness_good = DataQualityStandards.get_threshold('timeliness_days', 'good')
        self.assertEqual(timeliness_good, 14)
    
    def test_required_fields(self):
        """Test required fields for reporting standards."""
        fbi_fields = DataQualityStandards.get_required_fields('FBI_UCR')
        self.assertIn('incident_id', fbi_fields)
        self.assertIn('date', fbi_fields)
        self.assertIn('crime_type', fbi_fields)
    
    def test_freshness_validation(self):
        """Test data freshness validation."""
        result = self.standards.validate_data_freshness(self.df)
        
        self.assertIn('valid', result)
        self.assertIn('quality_level', result)
        self.assertIn('days_old', result)
        self.assertTrue(result['valid'])  # Recent data
    
    def test_completeness_validation(self):
        """Test data completeness validation."""
        result = self.standards.validate_completeness(self.df)
        
        self.assertIn('completeness_score', result)
        self.assertIn('missing_cells', result)
        self.assertGreater(result['completeness_score'], 0.8)  # Mostly complete
    
    def test_required_fields_validation(self):
        """Test required fields validation."""
        result = self.standards.validate_required_fields(self.df)
        
        self.assertIn('valid', result)
        self.assertIn('missing_fields', result)
        self.assertTrue(result['valid'])  # Has all FBI_UCR fields
    
    def test_quality_score_calculation(self):
        """Test comprehensive quality score calculation."""
        score = self.standards.calculate_quality_score(self.df)
        
        self.assertIn('overall_score', score)
        self.assertIn('overall_quality', score)
        self.assertIn('freshness', score)
        self.assertIn('completeness', score)
        self.assertIn('field_compliance', score)
        self.assertGreater(score['overall_score'], 0)


class TestCrimeDataIngestion(unittest.TestCase):
    """Test enhanced CrimeDataIngestion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        DataSource.clear_registry()
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_csv = self.test_dir / "test_data.csv"
        
        df = pd.DataFrame({
            'incident_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'crime_type': ['Theft', 'Assault', 'Burglary']
        })
        df.to_csv(self.test_csv, index=False)
        
        self.ingestion = CrimeDataIngestion()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        DataSource.clear_registry()
    
    def test_factory_methods(self):
        """Test factory methods for creating sources."""
        csv_source = self.ingestion.create_csv_source(str(self.test_csv))
        self.assertIsInstance(csv_source, CSVCrimeDataSource)
        
        api_source = self.ingestion.create_api_source("https://api.example.com/data")
        self.assertIsInstance(api_source, APICrimeDataSource)
    
    def test_load_from_source(self):
        """Test loading data from a source."""
        source = self.ingestion.create_csv_source(str(self.test_csv))
        df = self.ingestion.load_from_source(source)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
    
    def test_ingestion_statistics(self):
        """Test ingestion statistics tracking."""
        source = self.ingestion.create_csv_source(str(self.test_csv))
        self.ingestion.load_from_source(source)
        
        stats = self.ingestion.get_ingestion_statistics()
        
        self.assertEqual(stats['total_loads'], 1)
        self.assertEqual(stats['successful_loads'], 1)
        self.assertEqual(stats['failed_loads'], 0)
        self.assertEqual(stats['success_rate'], 1.0)
    
    def test_cache_management(self):
        """Test data caching functionality."""
        source = self.ingestion.create_csv_source(str(self.test_csv))
        df = self.ingestion.load_from_source(source, source_name="test_data")
        
        # Retrieve from cache
        cached = self.ingestion.get_loaded_data("test_data")
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), len(df))
        
        # List cached
        cached_names = self.ingestion.list_cached_data()
        self.assertIn("test_data", cached_names)
        
        # Clear cache
        self.ingestion.clear_cache("test_data")
        self.assertIsNone(self.ingestion.get_loaded_data("test_data"))


class TestCrimeDataStorageUtils(unittest.TestCase):
    """Test enhanced CrimeDataStorageUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage = CrimeDataStorageUtils(base_output_dir=str(self.test_dir))
        
        self.df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_save_to_csv(self):
        """Test saving DataFrame to CSV with tracking."""
        filepath = self.test_dir / "output.csv"
        saved_path = self.storage.save_to_csv(self.df, str(filepath))
        
        self.assertTrue(saved_path.exists())
        
        # Check file registry
        files = self.storage.list_saved_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]['file_type'], 'csv')
    
    def test_timestamped_filename(self):
        """Test timestamped filename generation."""
        filepath = self.test_dir / "output.csv"
        saved_path = self.storage.save_to_csv(self.df, str(filepath), use_timestamp=True)
        
        # Should contain timestamp
        self.assertIn('_', saved_path.name)
        self.assertTrue(saved_path.exists())
    
    def test_file_versioning(self):
        """Test file versioning system."""
        filepath = self.test_dir / "output.csv"
        
        # Save first version
        self.storage.save_to_csv(self.df, str(filepath))
        
        # Save again (should create version)
        self.storage.save_to_csv(self.df, str(filepath))
        
        # Check versions
        versions = self.storage.get_file_versions(str(filepath))
        self.assertGreater(len(versions), 0)
    
    def test_storage_statistics(self):
        """Test storage statistics tracking."""
        filepath = self.test_dir / "output.csv"
        self.storage.save_to_csv(self.df, str(filepath))
        
        stats = self.storage.get_storage_statistics()
        
        self.assertEqual(stats['total_saves'], 1)
        self.assertEqual(stats['successful_saves'], 1)
        self.assertEqual(stats['failed_saves'], 0)
        self.assertGreater(stats['total_bytes_written'], 0)
    
    def test_checksum_verification(self):
        """Test checksum calculation and verification."""
        filepath = self.test_dir / "output.csv"
        saved_path = self.storage.save_to_csv(self.df, str(filepath))
        
        # Load with verification
        df_loaded = self.storage.load_from_csv(str(saved_path), verify_checksum=True)
        self.assertEqual(len(df_loaded), len(self.df))


class TestPipelineManager(unittest.TestCase):
    """Test PipelineManager orchestration."""
    
    def setUp(self):
        """Set up test fixtures."""
        DataSource.clear_registry()
        DataProcessor.clear_registry()
        
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_csv = self.test_dir / "test_data.csv"
        
        df = pd.DataFrame({
            'incident_id': [1, 2, 3, 4, 5],
            'date': pd.date_range('2024-12-01', periods=5),
            'crime_type': ['Theft', 'Assault', 'Burglary', 'Robbery', 'Vandalism'],
            'location': ['A', 'B', 'C', 'D', 'E'],
            'value': [100, 200, 300, 400, 500],
            'count': [1, 2, 3, 4, 5]
        })
        df.to_csv(self.test_csv, index=False)
        
        self.pipeline = PipelineManager(verbose=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        DataSource.clear_registry()
        DataProcessor.clear_registry()
    
    def test_factory_methods(self):
        """Test pipeline factory methods."""
        csv_source = self.pipeline.create_csv_source(str(self.test_csv))
        self.assertIsInstance(csv_source, CSVCrimeDataSource)
        
        # Load data first
        self.pipeline.load_data(csv_source)
        
        cleaner = self.pipeline.create_cleaner()
        self.assertIsInstance(cleaner, CrimeDataCleaner)
        
        transformer = self.pipeline.create_transformer()
        self.assertIsInstance(transformer, CrimeDataTransformation)
        
        analyzer = self.pipeline.create_analyzer()
        self.assertIsInstance(analyzer, CrimeDataAnalysis)
    
    def test_fluent_interface(self):
        """Test method chaining."""
        result = (self.pipeline
                 .load_data(str(self.test_csv))
                 .clean(strategy='mean')
                 .transform(generate_features=True)
                 .analyze())
        
        self.assertIsInstance(result, PipelineManager)
        self.assertIsNotNone(self.pipeline.data)
    
    def test_processing_schedule(self):
        """Test processing schedule analytics."""
        self.pipeline.load_data(str(self.test_csv))
        self.pipeline.clean()
        
        schedule = self.pipeline.get_processing_schedule()
        
        self.assertGreater(len(schedule), 0)
        self.assertTrue(any(s['stage'] == 'ingestion' for s in schedule))
        self.assertTrue(any(s['stage'] == 'cleaning' for s in schedule))
    
    def test_data_quality_report(self):
        """Test data quality reporting."""
        standards = DataQualityStandards("Maryland", "LOCAL")
        pipeline = PipelineManager(quality_standards=standards)
        
        pipeline.load_data(str(self.test_csv))
        
        report = pipeline.get_data_quality_report()
        
        self.assertIn('current_data_shape', report)
        self.assertIn('quality_assessment', report)
    
    def test_pipeline_lineage(self):
        """Test pipeline lineage tracking."""
        self.pipeline.load_data(str(self.test_csv))
        self.pipeline.clean()
        self.pipeline.transform()
        
        lineage = self.pipeline.get_pipeline_lineage()
        
        self.assertIn('pipeline_id', lineage)
        self.assertIn('processors', lineage)
        self.assertGreater(len(lineage['processors']), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        self.pipeline.load_data(str(self.test_csv))
        self.pipeline.clean()
        
        metrics = self.pipeline.get_performance_metrics()
        
        self.assertIn('ingestion', metrics)
        self.assertIn('storage', metrics)
        self.assertIn('processors', metrics)
    
    def test_pipeline_validation(self):
        """Test pipeline validation."""
        self.pipeline.load_data(str(self.test_csv))
        
        validation = self.pipeline.validate_pipeline()
        
        self.assertIn('valid', validation)
        self.assertIn('issues', validation)
        self.assertIn('warnings', validation)
    
    def test_export_functionality(self):
        """Test pipeline state export."""
        self.pipeline.load_data(str(self.test_csv))
        
        state = self.pipeline.export_pipeline_state()
        
        self.assertIn('pipeline_id', state)
        self.assertIn('components', state)
        self.assertIn('processors', state)
        
        config = self.pipeline.export_pipeline_config()
        
        self.assertIn('pipeline_type', config)
        self.assertIn('version', config)
    
    def test_complete_pipeline(self):
        """Test running the complete pipeline."""
        output_file = self.test_dir / "results.csv"
        
        result = self.pipeline.run_full_pipeline(
            source=str(self.test_csv),
            clean_strategy='mean',
            scale_columns=['value', 'count'],
            output_file=str(output_file)
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(any(output_file.parent.glob('results*.csv')))


class TestPolymorphism(unittest.TestCase):
    """Test polymorphic behavior across the system."""
    
    def setUp(self):
        """Set up test fixtures."""
        DataSource.clear_registry()
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_csv = self.test_dir / "test.csv"
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        df.to_csv(self.test_csv, index=False)
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)
        DataSource.clear_registry()
    
    def test_source_polymorphism(self):
        """Test that different sources work polymorphically."""
        sources = [
            CSVCrimeDataSource(str(self.test_csv)),
        ]
        
        for source in sources:
            # All sources have these methods
            self.assertTrue(source.validate_source())
            self.assertIsNotNone(source.get_source_info())
            self.assertIsNotNone(source.metadata)
    
    def test_processor_polymorphism(self):
        """Test that different processors work polymorphically."""
        df = pd.DataFrame({'value': [1, 2, 3], 'count': [10, 20, 30]})
        
        processors = [
            CrimeDataAnalysis(df),
            CrimeDataCleaner(df),
            CrimeDataTransformation(df)
        ]
        
        for processor in processors:
            # All processors have these methods
            self.assertTrue(callable(processor.process))
            self.assertTrue(callable(processor.validate))
            self.assertIsNotNone(processor.get_summary())


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedDataSource))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityStandards))
    suite.addTests(loader.loadTestsFromTestCase(TestCrimeDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestCrimeDataStorageUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineManager))
    suite.addTests(loader.loadTestsFromTestCase(TestPolymorphism))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)