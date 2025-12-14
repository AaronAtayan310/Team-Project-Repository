"""
Crime Research Data Pipeline - Enhanced Data Ingestion

This module defines the enhanced CrimeDataIngestion class with advanced features
including source registry integration, caching, data validation, and
comprehensive ingestion tracking.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

import os
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid
from src.proj4_data_source import DataSource
from src.proj4_specialized_sources import CSVCrimeDataSource, APICrimeDataSource, DatabaseCrimeDataSource
from src.proj4_data_quality_standards import DataQualityStandards


class CrimeDataIngestion:
    """
    Enhanced class for core data ingestion features in the data pipeline.
    
    Demonstrates:
    - UUID-based identification
    - Integration with DataSource registry
    - Quality validation on ingestion
    - Enhanced caching with metadata
    - Ingestion statistics and reporting
    - Factory methods for creating sources
    """
    
    def __init__(self, default_timeout: int = 10, track_sources: bool = True, quality_standards: Optional[DataQualityStandards] = None):
        """
        Initialize an enhanced CrimeDataIngestion object.
        
        Args:
            default_timeout: Default timeout for API requests
            track_sources: Whether to track loaded sources
            quality_standards: Optional DataQualityStandards for validation
        """
        if not isinstance(default_timeout, int):
            raise TypeError("default_timeout must be an integer")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be a positive integer")
        if not isinstance(track_sources, bool):
            raise TypeError("track_sources must be a boolean")
        
        # UUID identification
        self._ingestion_id = str(uuid.uuid4())
        self._created_at = datetime.now()
        
        # Configuration
        self._default_timeout = default_timeout
        self._track_sources = track_sources
        self._quality_standards = quality_standards
        
        # Tracking
        self._data_sources: List[Dict[str, Any]] = []  # Legacy tracking
        self._loaded_data: Dict[str, pd.DataFrame] = {}  # Cache loaded DataFrames
        self._ingestion_history: List[Dict[str, Any]] = []  # Detailed ingestion log
        
        # Statistics
        self._total_loads = 0
        self._successful_loads = 0
        self._failed_loads = 0
    
    @property
    def ingestion_id(self) -> str:
        """Get the unique identifier for this ingestion instance."""
        return self._ingestion_id
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    @property
    def default_timeout(self) -> int:
        """Get default timeout for API requests."""
        return self._default_timeout
    
    @default_timeout.setter
    def default_timeout(self, value: int):
        """Set the default timeout for API requests."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("default_timeout must be a positive integer")
        self._default_timeout = value
    
    def create_csv_source(self, filepath: str, name: Optional[str] = None, **kwargs) -> CSVCrimeDataSource:
        """
        Factory method to create a CSVCrimeDataSource.
        
        Args:
            filepath: Path to CSV file
            name: Optional name for the source
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            CSVCrimeDataSource: Created source instance
        """
        source = CSVCrimeDataSource(filepath, name=name, **kwargs)
        return source
    
    def create_api_source(self, url: str, params: Optional[Dict] = None, timeout: Optional[int] = None, name: Optional[str] = None) -> APICrimeDataSource:
        """
        Factory method to create an APICrimeDataSource.
        
        Args:
            url: API endpoint URL
            params: Optional query parameters
            timeout: Optional timeout (uses default if not provided)
            name: Optional name for the source
            
        Returns:
            APICrimeDataSource: Created source instance
        """
        actual_timeout = timeout if timeout is not None else self._default_timeout
        source = APICrimeDataSource(url, params=params, timeout=actual_timeout, name=name)
        return source
    
    def create_database_source(self, connection_string: str, query: str, name: Optional[str] = None) -> DatabaseCrimeDataSource:
        """
        Factory method to create a DatabaseCrimeDataSource.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            name: Optional name for the source
            
        Returns:
            DatabaseCrimeDataSource: Created source instance
        """
        source = DatabaseCrimeDataSource(connection_string, query, name=name)
        return source
    
    def load_from_source(self, source: DataSource, 
                        source_name: Optional[str] = None,
                        validate_quality: bool = True) -> pd.DataFrame:
        """
        Load data using a DataSource object with enhanced tracking and validation.
        
        Args:
            source: A DataSource object (CSV, API, Database, etc)
            source_name: Optional name to cache the loaded data
            validate_quality: Whether to validate data quality after loading
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            TypeError: If source is not a DataSource instance
            ValueError: If source validation fails
        """
        # Validate source type
        if not isinstance(source, DataSource):
            if not hasattr(source, 'load') or not hasattr(source, 'validate_source'):
                raise TypeError(
                    "source must be a DataSource object with load() and validate_source() methods"
                )
        
        # Track ingestion start
        ingestion_record = {
            'ingestion_id': str(uuid.uuid4()),
            'source_id': source.source_id if hasattr(source, 'source_id') else 'unknown',
            'source_type': source.__class__.__name__,
            'source_name': source_name or source.name if hasattr(source, 'name') else 'unnamed',
            'started_at': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        self._total_loads += 1
        
        try:
            # Validate source before loading
            if not source.validate_source():
                raise ValueError(f"Invalid data source: {source}")
            
            # Load data (polymorphic call)
            df = source.load()
            
            # Validate data quality if requested and standards available
            quality_result = None
            if validate_quality and self._quality_standards:
                quality_result = self._quality_standards.calculate_quality_score(df)
            
            # Update ingestion record
            ingestion_record.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'success',
                'rows_loaded': len(df),
                'columns_loaded': len(df.columns),
                'quality_score': quality_result['overall_score'] if quality_result else None,
                'quality_level': quality_result['overall_quality'] if quality_result else None,
            })
            
            # Track the source (legacy tracking)
            if self._track_sources:
                self._data_sources.append({
                    'source_type': source.__class__.__name__,
                    'metadata': source.metadata if hasattr(source, 'metadata') else {},
                    'loaded_at': datetime.now().isoformat(),
                    'rows': len(df),
                    'columns': len(df.columns)
                })
            
            # Cache the data if name provided
            if source_name:
                self._loaded_data[source_name] = df
            
            self._successful_loads += 1
            
        except Exception as e:
            # Update ingestion record with error
            ingestion_record.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            self._failed_loads += 1
            raise
        
        finally:
            # Always record the ingestion attempt
            self._ingestion_history.append(ingestion_record)
        
        return df
    
    def load_csv(self, filepath: str, validate_quality: bool = True, **kwargs) -> pd.DataFrame:
        """
        Convenience method to load CSV using CSVCrimeDataSource.
        
        Args:
            filepath: Path to the CSV file
            validate_quality: Whether to validate data quality
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            pd.DataFrame: Loaded data
        """
        csv_source = self.create_csv_source(filepath, **kwargs)
        return self.load_from_source(csv_source, source_name=filepath, validate_quality=validate_quality)
    
    def fetch_api_data(self, url: str, params: Optional[Dict[str, Any]] = None, 
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from a REST API endpoint (legacy method).
        
        Args:
            url: The API URL
            params: Query parameters
            timeout: Request timeout
            
        Returns:
            dict: Parsed JSON response
        """
        if not isinstance(url, str):
            raise TypeError("URL must be a string")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary or None")
        
        actual_timeout = timeout if timeout is not None else self._default_timeout
        
        response = requests.get(url, params=params, timeout=actual_timeout)
        response.raise_for_status()
        data = response.json()
        
        if self._track_sources:
            self._data_sources.append({
                'type': 'api',
                'source': url,
                'params': params,
                'status_code': response.status_code
            })
        
        return data
    
    @staticmethod
    def validate_csv_path(file_path: str) -> bool:
        """
        Validate whether a given file path points to an existing CSV file.
        
        Args: 
            file_path: The actual file path used in validating a given csv file
            
        Returns:
            bool: True if valid CSV path
        """
        if not isinstance(file_path, str):
            raise TypeError("File path must be a string")
        return os.path.isfile(file_path) and file_path.lower().endswith(".csv")
    
    def get_loaded_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve previously loaded data from cache.
        
        Args:
            source_name: Name of the cached data source
            
        Returns:
            Optional[pd.DataFrame]: Cached DataFrame or None
        """
        return self._loaded_data.get(source_name)
    
    def list_cached_data(self) -> List[str]:
        """
        Get a list of all cached data source names.
        
        Returns:
            List[str]: Names of cached data sources
        """
        return list(self._loaded_data.keys())
    
    def clear_cache(self, source_name: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            source_name: Optional specific source to clear (if None, clears all)
        """
        if source_name:
            self._loaded_data.pop(source_name, None)
        else:
            self._loaded_data.clear()
    
    def clear_sources(self):
        """Clear the list of tracked data sources and cached data (legacy method)."""
        self._data_sources.clear()
        self._loaded_data.clear()
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive ingestion statistics.
        
        Returns:
            Dict: Ingestion statistics
        """
        return {
            'ingestion_id': self._ingestion_id,
            'created_at': self._created_at.isoformat(),
            'total_loads': self._total_loads,
            'successful_loads': self._successful_loads,
            'failed_loads': self._failed_loads,
            'success_rate': self._successful_loads / self._total_loads if self._total_loads > 0 else 0.0,
            'cached_datasets': len(self._loaded_data),
            'tracked_sources': len(self._data_sources),
        }
    
    def get_ingestion_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get ingestion history.
        
        Args:
            limit: Optional limit on number of records to return
            
        Returns:
            List[Dict]: Ingestion history records
        """
        if limit:
            return self._ingestion_history[-limit:]
        return self._ingestion_history.copy()
    
    def get_source_summary(self) -> Dict[str, Any]:
        """
        Get summary of all sources accessed via this ingestion instance.
        
        Returns:
            Dict: Source summary with registry integration
        """
        # Get sources from DataSource registry
        all_sources = DataSource.get_all_sources()
        
        by_type = {}
        for source in all_sources:
            source_type = source.__class__.__name__
            by_type[source_type] = by_type.get(source_type, 0) + 1
        
        return {
            'total_registered_sources': len(all_sources),
            'by_type': by_type,
            'cached_data': len(self._loaded_data),
            'cache_names': list(self._loaded_data.keys())
        }
    
    def __str__(self) -> str:
        """User-friendly representation."""
        sources_count = len(self._data_sources)
        cached_count = len(self._loaded_data)
        tracking_status = "enabled" if self._track_sources else "disabled"
        return (
            f"CrimeDataIngestion (id={self._ingestion_id[:8]}, timeout={self._default_timeout}s, "
            f"tracking={tracking_status}, loads={self._successful_loads}/{self._total_loads}, "
            f"cached={cached_count})"
        )
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"CrimeDataIngestion(ingestion_id='{self._ingestion_id[:8]}...', "
            f"successful_loads={self._successful_loads}, "
            f"failed_loads={self._failed_loads})"
        )
