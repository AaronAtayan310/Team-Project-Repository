"""
Crime Research Data Pipeline - Enhanced Data Source Base Class

This module defines the finalized base class for all data sources with
advanced features including UUID tracking, temporal metadata, registry
pattern, and comprehensive source management capabilities. Refactored
version of the base class seen in proj3_base_classes.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional, ClassVar
from datetime import datetime
from pathlib import Path
import hashlib
import uuid


class DataSource(ABC):
    """
    Enhanced base class for all data source types.
    
    Demonstrates:
    - Concrete base class (not purely abstract)
    - Instance registry for tracking all sources
    - Temporal tracking of all operations
    - Rich metadata beyond basic info
    """
    
    # Class-level registry for tracking all source instances
    _registry: ClassVar[Dict[str, 'DataSource']] = {}
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the enhanced DataSource.
        
        Args:
            name: Optional human-readable name for this source
        """
        # UUID-based identification
        self._source_id = str(uuid.uuid4())
        self._name = name or f"{self.__class__.__name__}_{self._source_id[:8]}"
        
        # Temporal tracking
        self._created_at = datetime.now()
        self._last_loaded_at: Optional[datetime] = None
        self._load_count = 0
        
        # Enhanced metadata
        self._source_metadata: Dict[str, Any] = {
            'source_id': self._source_id,
            'source_type': self.__class__.__name__,
            'name': self._name,
            'created_at': self._created_at.isoformat(),
        }
        
        # Load history tracking
        self._load_history: List[Dict[str, Any]] = []
        
        # Data lineage tracking
        self._data_checksum: Optional[str] = None
        self._last_data_shape: Optional[tuple] = None
        
        # Register this instance
        DataSource._registry[self._source_id] = self
    
    @property
    def source_id(self) -> str:
        """Get the unique identifier for this source."""
        return self._source_id
    
    @property
    def name(self) -> str:
        """Get the human-readable name for this source."""
        return self._name
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    @property
    def last_loaded_at(self) -> Optional[datetime]:
        """Get the timestamp of the last load operation."""
        return self._last_loaded_at
    
    @property
    def load_count(self) -> int:
        """Get the number of times data has been loaded from this source."""
        return self._load_count
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source."""
        return self._source_metadata.copy()
    
    @property
    def load_history(self) -> List[Dict[str, Any]]:
        """Get the history of load operations."""
        return self._load_history.copy()
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data from the source.
        
        Must be implemented by all subclasses.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate the data source before loading.
        
        Must be implemented by all subclasses.
        
        Returns:
            bool: True if source is valid
        """
        pass
    
    def _track_load(self, df: pd.DataFrame, duration: float, success: bool = True):
        """
        Track a load operation with temporal and quality metadata.
        
        Args:
            df: The loaded DataFrame
            duration: Time taken to load (seconds)
            success: Whether the load was successful
        """
        self._last_loaded_at = datetime.now()
        if success:
            self._load_count += 1
            self._last_data_shape = df.shape
            self._data_checksum = self._calculate_checksum(df)
        
        # Record in load history
        load_record = {
            'timestamp': self._last_loaded_at.isoformat(),
            'success': success,
            'duration_seconds': duration,
            'rows_loaded': df.shape[0] if success else 0,
            'columns_loaded': df.shape[1] if success else 0,
            'data_checksum': self._data_checksum if success else None,
        }
        self._load_history.append(load_record)
        
        # Update metadata
        self._source_metadata['last_loaded_at'] = self._last_loaded_at.isoformat()
        self._source_metadata['load_count'] = self._load_count
        self._source_metadata['last_data_shape'] = self._last_data_shape
    
    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """
        Calculate a checksum for data lineage tracking.
        
        Args:
            df: DataFrame to checksum
            
        Returns:
            str: MD5 checksum of the DataFrame
        """
        # Create a string representation of the DataFrame
        data_str = f"{df.shape}_{df.columns.tolist()}_{len(df)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive load statistics for this source.
        
        Returns:
            Dict: Load statistics including timing, success rates, etc.
        """
        if not self._load_history:
            return {
                'total_loads': 0,
                'successful_loads': 0,
                'failed_loads': 0,
                'success_rate': 0.0,
                'average_duration': 0.0,
                'total_rows_loaded': 0,
            }
        
        successful = [h for h in self._load_history if h['success']]
        failed = [h for h in self._load_history if not h['success']]
        
        return {
            'total_loads': len(self._load_history),
            'successful_loads': len(successful),
            'failed_loads': len(failed),
            'success_rate': len(successful) / len(self._load_history) if self._load_history else 0.0,
            'average_duration': sum(h['duration_seconds'] for h in successful) / len(successful) if successful else 0.0,
            'total_rows_loaded': sum(h['rows_loaded'] for h in successful),
            'last_load_timestamp': self._last_loaded_at.isoformat() if self._last_loaded_at else None,
        }
    
    def get_source_info(self) -> str:
        """
        Get a comprehensive string representation of the source info.
        
        Returns:
            str: Detailed source information
        """
        return (
            f"{self.__class__.__name__}(id={self._source_id[:8]}, "
            f"name='{self._name}', loads={self._load_count}, "
            f"created={self._created_at.strftime('%Y-%m-%d %H:%M:%S')})"
        )
    
    def get_data_lineage(self) -> Dict[str, Any]:
        """
        Get data lineage information for tracking data provenance.
        
        Returns:
            Dict: Lineage information including checksums, timestamps, shapes
        """
        return {
            'source_id': self._source_id,
            'source_name': self._name,
            'source_type': self.__class__.__name__,
            'created_at': self._created_at.isoformat(),
            'last_loaded_at': self._last_loaded_at.isoformat() if self._last_loaded_at else None,
            'load_count': self._load_count,
            'last_data_checksum': self._data_checksum,
            'last_data_shape': self._last_data_shape,
        }
    
    @classmethod
    def get_source_by_id(cls, source_id: str) -> Optional['DataSource']:
        """
        Retrieve a source instance by its UUID.
        
        Args:
            source_id: The UUID of the source
            
        Returns:
            DataSource instance or None if not found
        """
        return cls._registry.get(source_id)
    
    @classmethod
    def get_all_sources(cls) -> List['DataSource']:
        """
        Get all registered source instances.
        
        Returns:
            List of all DataSource instances
        """
        return list(cls._registry.values())
    
    @classmethod
    def get_sources_by_type(cls, source_type: str) -> List['DataSource']:
        """
        Get all sources of a specific type.
        
        Args:
            source_type: The class name of the source type
            
        Returns:
            List of matching DataSource instances
        """
        return [
            source for source in cls._registry.values()
            if source.__class__.__name__ == source_type
        ]
    
    @classmethod
    def clear_registry(cls):
        """Clear the source registry (useful for testing)."""
        cls._registry.clear()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}(source_id='{self._source_id[:8]}...', "
            f"name='{self._name}', load_count={self._load_count})"
        )
    
    def __str__(self) -> str:
        """User-friendly representation."""
        status = f"loaded {self._load_count} times" if self._load_count > 0 else "not yet loaded"
        return f"{self._name} ({self.__class__.__name__}): {status}"
