"""
Crime Research Data Pipeline - Pipeline Manager

This module defines the PipelineManager class which orchestrates the entire
crime data pipeline with advanced features including factory methods, analytics,
task management, and comprehensive reporting.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

from .proj4_data_source import DataSource
from .proj4_data_processor import DataProcessor
from .proj4_specialized_sources import CSVCrimeDataSource, APICrimeDataSource, DatabaseCrimeDataSource
from .proj4_specialized_processors import CrimeDataAnalysis, CrimeDataCleaner, CrimeDataTransformation
from .proj4_data_ingestion import CrimeDataIngestion
from .proj4_data_utilities import CrimeDataStorageUtils
from .proj4_data_quality_standards import DataQualityStandards, QualityLevel

import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import uuid
import time


class PipelineManager:
    """
    Enhanced orchestrator for the complete crime data pipeline.
    
    Demonstrates:
    - UUID-based identification
    - Comprehensive pipeline analytics
    - Data lineage tracking
    - Quality reporting across all stages
    - Performance monitoring
    - Pipeline state export/import
    """
    
    def __init__(self, ingestion: Optional[CrimeDataIngestion] = None, storage: Optional[CrimeDataStorageUtils] = None, quality_standards: Optional[DataQualityStandards] = None, verbose: bool = False):
        """
        Initialize the PipelineManager with optional components.
        
        Args:
            ingestion: Optional CrimeDataIngestion instance
            storage: Optional CrimeDataStorageUtils instance
            quality_standards: Optional DataQualityStandards instance
            verbose: Whether to print detailed information
        """
        # UUID identification
        self._pipeline_id = str(uuid.uuid4())
        self._created_at = datetime.now()
        
        # Core components (COMPOSITION - HAS-A relationships)
        self.ingestion = ingestion or CrimeDataIngestion()
        self.storage = storage or CrimeDataStorageUtils()
        self.quality_standards = quality_standards
        
        # Current pipeline state
        self.data: Optional[pd.DataFrame] = None
        self._current_source: Optional[DataSource] = None
        
        # Dynamic processors (created on demand)
        self._cleaner: Optional[CrimeDataCleaner] = None
        self._transformer: Optional[CrimeDataTransformation] = None
        self._analyzer: Optional[CrimeDataAnalysis] = None
        
        # Configuration
        self._verbose = verbose
        
        # Pipeline tracking
        self._pipeline_history: List[Dict[str, Any]] = []
        self._pipeline_steps = 0
        
        # Log initialization
        self._log_pipeline_step("Pipeline initialized", "success")
    
    @property
    def pipeline_id(self) -> str:
        """Get the unique identifier for this pipeline."""
        return self._pipeline_id
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    def create_csv_source(self, filepath: str, name: Optional[str] = None,
                         **kwargs) -> CSVCrimeDataSource:
        """
        Factory method to create a CSVCrimeDataSource.
        
        Args:
            filepath: Path to CSV file
            name: Optional name for the source
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            CSVCrimeDataSource: Created source instance
        """
        return self.ingestion.create_csv_source(filepath, name=name, **kwargs)
    
    def create_api_source(self, url: str, params: Optional[Dict] = None,
                         timeout: Optional[int] = None,
                         name: Optional[str] = None) -> APICrimeDataSource:
        """
        Factory method to create an APICrimeDataSource.
        
        Args:
            url: API endpoint URL
            params: Optional query parameters
            timeout: Optional timeout
            name: Optional name for the source
            
        Returns:
            APICrimeDataSource: Created source instance
        """
        return self.ingestion.create_api_source(url, params=params, timeout=timeout, name=name)
    
    def create_database_source(self, connection_string: str, query: str,
                              name: Optional[str] = None) -> DatabaseCrimeDataSource:
        """
        Factory method to create a DatabaseCrimeDataSource.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            name: Optional name for the source
            
        Returns:
            DatabaseCrimeDataSource: Created source instance
        """
        return self.ingestion.create_database_source(connection_string, query, name=name)
    
    def create_cleaner(self, df: Optional[pd.DataFrame] = None) -> CrimeDataCleaner:
        """
        Factory method to create a CrimeDataCleaner.
        
        Args:
            df: Optional DataFrame (uses pipeline data if None)
            
        Returns:
            CrimeDataCleaner: Created cleaner instance
        """
        data = df if df is not None else self.data
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        self._cleaner = CrimeDataCleaner(
            data,
            verbose=self._verbose,
            quality_standards=self.quality_standards
        )
        return self._cleaner
    
    def create_transformer(self, df: Optional[pd.DataFrame] = None) -> CrimeDataTransformation:
        """
        Factory method to create a CrimeDataTransformation.
        
        Args:
            df: Optional DataFrame (uses pipeline data if None)
            
        Returns:
            CrimeDataTransformation: Created transformer instance
        """
        data = df if df is not None else self.data
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        self._transformer = CrimeDataTransformation(
            data,
            verbose=self._verbose
        )
        return self._transformer
    
    def create_analyzer(self, df: Optional[pd.DataFrame] = None) -> CrimeDataAnalysis:
        """
        Factory method to create a CrimeDataAnalysis.
        
        Args:
            df: Optional DataFrame (uses pipeline data if None)
            
        Returns:
            CrimeDataAnalysis: Created analyzer instance
        """
        data = df if df is not None else self.data
        if data is None:
            raise ValueError("No data available. Load data first.")
        
        self._analyzer = CrimeDataAnalysis(
            data,
            verbose=self._verbose,
            quality_standards=self.quality_standards
        )
        return self._analyzer
    
    def load_data(self, source: Union[DataSource, str],
                 source_name: Optional[str] = None,
                 validate_quality: bool = True) -> 'PipelineManager':
        """
        Load data from a source into the pipeline.
        
        Args:
            source: DataSource instance or filepath string
            source_name: Optional name for caching
            validate_quality: Whether to validate data quality
            
        Returns:
            PipelineManager: Self for method chaining
        """
        start_time = time.time()
        
        try:
            # Handle string filepath
            if isinstance(source, str):
                source = self.create_csv_source(source)
            
            # Load using ingestion component
            self.data = self.ingestion.load_from_source(
                source,
                source_name=source_name,
                validate_quality=validate_quality
            )
            self._current_source = source
            
            duration = time.time() - start_time
            
            self._log_pipeline_step(
                f"Loaded data from {source.__class__.__name__}",
                "success",
                {
                    'source_id': source.source_id if hasattr(source, 'source_id') else 'unknown',
                    'rows': len(self.data),
                    'columns': len(self.data.columns),
                    'duration': duration
                }
            )
            
            if self._verbose:
                print(f"✓ Data loaded: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            
        except Exception as e:
            self._log_pipeline_step(f"Failed to load data", "failed", {'error': str(e)})
            raise
        
        return self
    
    def clean(self, strategy: str = 'mean', 
             columns: Optional[List[str]] = None) -> 'PipelineManager':
        """
        Clean the pipeline data.
        
        Args:
            strategy: Cleaning strategy
            columns: Optional specific columns
            
        Returns:
            PipelineManager: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data to clean. Load data first.")
        
        start_time = time.time()
        
        try:
            # Create cleaner if not exists
            if self._cleaner is None:
                self.create_cleaner()
            
            # Perform cleaning
            self._cleaner.handle_missing_values(strategy=strategy, columns=columns)
            self.data = self._cleaner.frame
            
            duration = time.time() - start_time
            
            self._log_pipeline_step(
                f"Cleaned data using '{strategy}' strategy",
                "success",
                {'strategy': strategy, 'duration': duration}
            )
            
            if self._verbose:
                print(f"✓ Data cleaned using '{strategy}' strategy")
            
        except Exception as e:
            self._log_pipeline_step(f"Failed to clean data", "failed", {'error': str(e)})
            raise
        
        return self
    
    def transform(self, scale_columns: Optional[List[str]] = None, generate_features: bool = True) -> 'PipelineManager':
        """
        Transform the pipeline data.
        
        Args:
            scale_columns: Optional columns to scale
            generate_features: Whether to generate new features
            
        Returns:
            PipelineManager: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data to transform. Load data first.")
        
        start_time = time.time()
        
        try:
            # Create transformer if not exists
            if self._transformer is None:
                self.create_transformer()
            
            # Perform transformation
            if generate_features:
                self._transformer.generate_features()
            
            if scale_columns:
                self._transformer.scale_features(scale_columns)
            
            self.data = self._transformer.frame
            
            duration = time.time() - start_time
            
            self._log_pipeline_step(
                "Transformed data",
                "success",
                {
                    'scaled_columns': scale_columns,
                    'generated_features': generate_features,
                    'duration': duration
                }
            )
            
            if self._verbose:
                print(f"✓ Data transformed")
            
        except Exception as e:
            self._log_pipeline_step(f"Failed to transform data", "failed", {'error': str(e)})
            raise
        
        return self
    
    def analyze(self) -> 'PipelineManager':
        """
        Analyze the pipeline data.
        
        Returns:
            PipelineManager: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data to analyze. Load data first.")
        
        start_time = time.time()
        
        try:
            # Create analyzer if not exists
            if self._analyzer is None:
                self.create_analyzer()
            
            # Perform analysis
            self._analyzer.process()
            
            duration = time.time() - start_time
            
            self._log_pipeline_step(
                "Analyzed data",
                "success",
                {'duration': duration}
            )
            
            if self._verbose:
                print(f"✓ Data analyzed")
            
        except Exception as e:
            self._log_pipeline_step(f"Failed to analyze data", "failed", {'error': str(e)})
            raise
        
        return self
    
    def save_results(self, filepath: str, use_timestamp: bool = False, compress: bool = False) -> 'PipelineManager':
        """
        Save pipeline results to file.
        
        Args:
            filepath: Output filepath
            use_timestamp: Whether to add timestamp
            compress: Whether to compress
            
        Returns:
            PipelineManager: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data to save. Load data first.")
        
        start_time = time.time()
        
        try:
            # Save using storage component
            saved_path = self.storage.save_to_csv(
                self.data,
                filepath,
                use_timestamp=use_timestamp,
                compress=compress
            )
            
            duration = time.time() - start_time
            
            self._log_pipeline_step(
                f"Saved results to {saved_path}",
                "success",
                {'filepath': str(saved_path), 'duration': duration}
            )
            
            if self._verbose:
                print(f"✓ Results saved to: {saved_path}")
            
        except Exception as e:
            self._log_pipeline_step(f"Failed to save results", "failed", {'error': str(e)})
            raise
        
        return self
    
    def run_full_pipeline(self, source: Union[DataSource, str], clean_strategy: str = 'mean', scale_columns: Optional[List[str]] = None, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete pipeline: load → clean → transform → analyze → save.
        
        Args:
            source: Data source
            clean_strategy: Cleaning strategy
            scale_columns: Columns to scale
            output_file: Optional output filepath
            
        Returns:
            pd.DataFrame: Processed data
        """
        pipeline_start = time.time()
        
        # Execute pipeline
        self.load_data(source)
        self.clean(strategy=clean_strategy)
        self.transform(scale_columns=scale_columns)
        self.analyze()
        
        if output_file:
            self.save_results(output_file, use_timestamp=True)
        
        pipeline_duration = time.time() - pipeline_start
        
        self._log_pipeline_step(
            "Completed full pipeline",
            "success",
            {'total_duration': pipeline_duration}
        )
        
        if self._verbose:
            print(f"\n✓ Full pipeline completed in {pipeline_duration:.2f}s")
        
        return self.data
    
    def get_processing_schedule(self) -> List[Dict[str, Any]]:
        """
        Get a schedule of what processing operations will/have occurred.
        
        Returns:
            List[Dict]: Processing schedule
        """
        schedule = []
        
        # Ingestion
        if self._current_source:
            schedule.append({
                'stage': 'ingestion',
                'component': 'CrimeDataIngestion',
                'status': 'completed' if self.data is not None else 'pending',
                'source_type': self._current_source.__class__.__name__ if self._current_source else None,
            })
        
        # Cleaning
        if self._cleaner:
            schedule.append({
                'stage': 'cleaning',
                'component': 'CrimeDataCleaner',
                'status': 'completed',
                'operations': len(self._cleaner.processing_history),
            })
        
        # Transformation
        if self._transformer:
            schedule.append({
                'stage': 'transformation',
                'component': 'CrimeDataTransformation',
                'status': 'completed',
                'operations': len(self._transformer.processing_history),
            })
        
        # Analysis
        if self._analyzer:
            schedule.append({
                'stage': 'analysis',
                'component': 'CrimeDataAnalysis',
                'status': 'completed',
                'operations': len(self._analyzer.processing_history),
            })
        
        return schedule
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report across all pipeline stages.
        
        Returns:
            Dict: Quality report
        """
        if self.data is None:
            return {'error': 'No data loaded'}
        
        report = {
            'pipeline_id': self._pipeline_id,
            'generated_at': datetime.now().isoformat(),
            'current_data_shape': self.data.shape,
        }
        
        # Quality standards assessment
        if self.quality_standards:
            quality_score = self.quality_standards.calculate_quality_score(self.data)
            report['quality_assessment'] = quality_score
        
        # Cleaning impact
        if self._cleaner:
            report['cleaning_impact'] = self._cleaner.get_cleaning_impact()
        
        # Data quality metrics
        if self._analyzer:
            report['data_quality_metrics'] = self._analyzer.get_data_quality_metrics()
        
        return report
    
    def get_pipeline_lineage(self) -> Dict[str, Any]:
        """
        Trace complete data lineage through the pipeline.
        
        Returns:
            Dict: Lineage information
        """
        lineage = {
            'pipeline_id': self._pipeline_id,
            'created_at': self._created_at.isoformat(),
            'pipeline_steps': self._pipeline_steps,
        }
        
        # Source lineage
        if self._current_source and hasattr(self._current_source, 'get_data_lineage'):
            lineage['source'] = self._current_source.get_data_lineage()
        
        # Processing lineage
        processors = []
        
        if self._cleaner:
            processors.append({
                'processor_id': self._cleaner.processor_id,
                'processor_type': 'CrimeDataCleaner',
                'operations': len(self._cleaner.processing_history),
            })
        
        if self._transformer:
            processors.append({
                'processor_id': self._transformer.processor_id,
                'processor_type': 'CrimeDataTransformation',
                'operations': len(self._transformer.processing_history),
                'feature_lineage': self._transformer.get_feature_lineage(),
            })
        
        if self._analyzer:
            processors.append({
                'processor_id': self._analyzer.processor_id,
                'processor_type': 'CrimeDataAnalysis',
                'operations': len(self._analyzer.processing_history),
            })
        
        lineage['processors'] = processors
        
        return lineage
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics across all components.
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {
            'pipeline_id': self._pipeline_id,
            'total_pipeline_steps': self._pipeline_steps,
        }
        
        # Ingestion metrics
        metrics['ingestion'] = self.ingestion.get_ingestion_statistics()
        
        # Storage metrics
        metrics['storage'] = self.storage.get_storage_statistics()
        
        # Processor metrics
        processor_metrics = {}
        
        if self._cleaner:
            processor_metrics['cleaner'] = self._cleaner.get_performance_metrics()
        
        if self._transformer:
            processor_metrics['transformer'] = self._transformer.get_performance_metrics()
        
        if self._analyzer:
            processor_metrics['analyzer'] = self._analyzer.get_performance_metrics()
        
        metrics['processors'] = processor_metrics
        
        return metrics
    
    def get_pending_operations(self) -> List[str]:
        """
        Get list of pending operations (what hasn't been done yet).
        
        Returns:
            List[str]: Pending operations
        """
        pending = []
        
        if self.data is None:
            pending.append("Load data from source")
        else:
            if self._cleaner is None:
                pending.append("Clean data")
            if self._transformer is None:
                pending.append("Transform data")
            if self._analyzer is None:
                pending.append("Analyze data")
        
        return pending if pending else ["Pipeline complete - no pending operations"]
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate that the pipeline is ready to run.
        
        Returns:
            Dict: Validation results
        """
        issues = []
        warnings = []
        
        # Check data
        if self.data is None:
            issues.append("No data loaded")
        else:
            if self.data.empty:
                issues.append("Data is empty")
            if self.data.isnull().all().any():
                warnings.append("Some columns are entirely null")
        
        # Check quality standards
        if self.quality_standards is None:
            warnings.append("No quality standards configured")
        
        # Check components
        if self._cleaner and not self._cleaner.validate():
            warnings.append("Data cleaner validation failed")
        
        if self._transformer and not self._transformer.validate():
            warnings.append("Data transformer validation failed")
        
        if self._analyzer and not self._analyzer.validate():
            warnings.append("Data analyzer validation failed")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat(),
        }
    
    def export_pipeline_state(self) -> Dict[str, Any]:
        """
        Export complete pipeline state for backup/analysis.
        
        Returns:
            Dict: Complete pipeline state
        """
        state = {
            'pipeline_id': self._pipeline_id,
            'created_at': self._created_at.isoformat(),
            'exported_at': datetime.now().isoformat(),
            'pipeline_steps': self._pipeline_steps,
            'pipeline_history': self._pipeline_history.copy(),
            'has_data': self.data is not None,
            'data_shape': self.data.shape if self.data is not None else None,
        }
        
        # Component states
        state['components'] = {
            'ingestion': {
                'id': self.ingestion.ingestion_id,
                'statistics': self.ingestion.get_ingestion_statistics(),
            },
            'storage': {
                'id': self.storage.storage_id,
                'statistics': self.storage.get_storage_statistics(),
            },
            'quality_standards': {
                'configured': self.quality_standards is not None,
                'jurisdiction': self.quality_standards.jurisdiction if self.quality_standards else None,
            }
        }
        
        # Processor states
        processors = {}
        if self._cleaner:
            processors['cleaner'] = {
                'id': self._cleaner.processor_id,
                'operations': len(self._cleaner.processing_history),
            }
        if self._transformer:
            processors['transformer'] = {
                'id': self._transformer.processor_id,
                'operations': len(self._transformer.processing_history),
            }
        if self._analyzer:
            processors['analyzer'] = {
                'id': self._analyzer.processor_id,
                'operations': len(self._analyzer.processing_history),
            }
        
        state['processors'] = processors
        
        return state
    
    def export_pipeline_config(self) -> Dict[str, Any]:
        """
        Export pipeline configuration for recreation.
        
        Returns:
            Dict: Pipeline configuration
        """
        config = {
            'pipeline_type': 'CrimePipelineManager',
            'version': '4.0',
            'created_at': self._created_at.isoformat(),
            'configuration': {
                'verbose': self._verbose,
                'quality_standards_configured': self.quality_standards is not None,
            }
        }
        
        if self.quality_standards:
            config['quality_standards'] = {
                'jurisdiction': self.quality_standards.jurisdiction,
                'reporting_standard': self.quality_standards.reporting_standard.value,
            }
        
        return config
    
    def _log_pipeline_step(self, step_name: str, status: str, extra_info: Optional[Dict[str, Any]] = None):
        """
        Log a pipeline step with enhanced metadata.
        
        Args:
            step_name: Name of the step
            status: Status (success, failed, etc.)
            extra_info: Additional information
        """
        self._pipeline_steps += 1
        
        log_entry = {
            'step_number': self._pipeline_steps,
            'step_name': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
        }
        
        if extra_info:
            log_entry.update(extra_info)
        
        self._pipeline_history.append(log_entry)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the pipeline state.
        
        Returns:
            Dict: Pipeline summary
        """
        summary = {
            'pipeline_id': self._pipeline_id,
            'created_at': self._created_at.isoformat(),
            'pipeline_steps': self._pipeline_steps,
            'has_data': self.data is not None,
            'data_shape': self.data.shape if self.data is not None else None,
            'components_initialized': {
                'cleaner': self._cleaner is not None,
                'transformer': self._transformer is not None,
                'analyzer': self._analyzer is not None,
            },
            'pending_operations': self.get_pending_operations(),
        }
        
        return summary
    
    def __str__(self) -> str:
        """User-friendly representation."""
        data_status = f"{self.data.shape[0]}×{self.data.shape[1]}" if self.data is not None else "No data"
        return (
            f"PipelineManager (id={self._pipeline_id[:8]}, "
            f"steps={self._pipeline_steps}, data={data_status})"
        )
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"PipelineManager(pipeline_id='{self._pipeline_id[:8]}...', "
            f"pipeline_steps={self._pipeline_steps})"
        )
