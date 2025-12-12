"""
Crime Research Data Pipeline - Enhanced Data Processor Abstract Base Class

This module defines the enhanced abstract base class for all data processing
classes with advanced features including UUID tracking, temporal metadata,
performance monitoring, and comprehensive processing history. Refactored
version of AbstractDataProcessor as seen in proj3_base_classes.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, ClassVar
from datetime import datetime, timedelta
import uuid


class DataProcessor(ABC):
    """
    Enhanced abstract base class for all data processing classes.

    Demonstrates:
    - Instance registry for tracking all processors
    - Temporal tracking with operation durations
    - Performance metrics collection
    - Data quality score tracking
    """
    
    # Class-level registry for tracking all processor instances
    _registry: ClassVar[Dict[str, 'DataProcessor']] = {}
    
    def __init__(self, frame: pd.DataFrame, verbose: bool = False, name: Optional[str] = None):
        """
        Initialize the enhanced DataProcessor with a DataFrame.
        
        Args:
            frame: The DataFrame to process
            verbose: If True, print processing information
            name: Optional human-readable name for this processor
            
        Raises:
            TypeError: If frame is not a pandas DataFrame
            ValueError: If frame is empty
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # UUID-based identification
        self._processor_id = str(uuid.uuid4())
        self._name = name or f"{self.__class__.__name__}_{self._processor_id[:8]}"
        
        # Data management
        self._frame = frame.copy()
        self._original_frame = frame.copy()  # Keep original for comparison
        
        # Temporal tracking
        self._created_at = datetime.now()
        self._last_processed_at: Optional[datetime] = None
        self._processing_count = 0
        
        # Processing history with enhanced metadata
        self._processing_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._total_processing_time = timedelta()
        
        # Data quality tracking
        self._quality_scores: List[Dict[str, Any]] = []
        
        # Configuration
        self._verbose = verbose
        
        # Register this instance
        DataProcessor._registry[self._processor_id] = self
        
        # Log initialization
        self._log_operation("Processor initialized", duration=0.0)
    
    @property
    def processor_id(self) -> str:
        """Get the unique identifier for this processor."""
        return self._processor_id
    
    @property
    def name(self) -> str:
        """Get the human-readable name for this processor."""
        return self._name
    
    @property
    def frame(self) -> pd.DataFrame:
        """Get the current DataFrame."""
        return self._frame
    
    @frame.setter
    def frame(self, value: pd.DataFrame):
        """Set the DataFrame with validation."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Value must be a pandas DataFrame")
        self._frame = value
    
    @property
    def original_frame(self) -> pd.DataFrame:
        """Get the original DataFrame (before any processing)."""
        return self._original_frame.copy()
    
    @property
    def processing_history(self) -> List[Dict[str, Any]]:
        """Get the history of processing operations with enhanced metadata."""
        return [h.copy() for h in self._processing_history]
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    @property
    def last_processed_at(self) -> Optional[datetime]:
        """Get the timestamp of the last processing operation."""
        return self._last_processed_at
    
    @property
    def processing_count(self) -> int:
        """Get the number of processing operations performed."""
        return self._processing_count
    
    def _log_operation(self, operation: str, duration: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        """
        Log operation to history with enhanced temporal and quality metadata.
        
        Args:
            operation: Description of operation
            duration: Time taken for the operation (seconds)
            metadata: Additional metadata about the operation
        """
        timestamp = datetime.now()
        
        # Create log entry with enhanced metadata
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'frame_shape': self._frame.shape,
            'processor_type': self.__class__.__name__,
        }
        
        # Add optional metadata
        if metadata:
            log_entry.update(metadata)
        
        self._processing_history.append(log_entry)
        
        # Update temporal tracking
        self._last_processed_at = timestamp
        self._total_processing_time += timedelta(seconds=duration)
        
        # Verbose output
        if self._verbose:
            print(f"[{self.__class__.__name__}] {operation} ({duration:.3f}s)")
    
    @abstractmethod
    def process(self) -> 'DataProcessor':
        """
        Perform the main processing operation.
        
        Must be implemented by all subclasses.
        
        Returns:
            DataProcessor: Self for method chaining
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the DataFrame state.
        
        Must be implemented by all subclasses.
        
        Returns:
            bool: True if validation passes
        """
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current DataFrame state.
        
        Returns:
            Dict: Summary statistics with enhanced metadata
        """
        return {
            'processor_id': self._processor_id,
            'processor_name': self._name,
            'processor_type': self.__class__.__name__,
            'shape': self._frame.shape,
            'columns': list(self._frame.columns),
            'dtypes': self._frame.dtypes.to_dict(),
            'missing_values': self._frame.isnull().sum().to_dict(),
            'operations_count': len(self._processing_history),
            'processing_count': self._processing_count,
            'total_processing_time_seconds': self._total_processing_time.total_seconds(),
            'created_at': self._created_at.isoformat(),
            'last_processed_at': self._last_processed_at.isoformat() if self._last_processed_at else None,
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for this processor.
        
        Returns:
            Dict: Performance metrics including timing, operation counts, etc
        """
        if not self._processing_history:
            return {
                'total_operations': 0,
                'total_time_seconds': 0.0,
                'average_operation_time': 0.0,
                'fastest_operation': None,
                'slowest_operation': None,
            }
        
        durations = [h['duration_seconds'] for h in self._processing_history]
        operations = [h['operation'] for h in self._processing_history]
        
        fastest_idx = durations.index(min(durations))
        slowest_idx = durations.index(max(durations))
        
        return {
            'total_operations': len(self._processing_history),
            'total_time_seconds': self._total_processing_time.total_seconds(),
            'average_operation_time': sum(durations) / len(durations),
            'fastest_operation': {
                'operation': operations[fastest_idx],
                'duration': durations[fastest_idx]
            },
            'slowest_operation': {
                'operation': operations[slowest_idx],
                'duration': durations[slowest_idx]
            },
            'operations_per_second': len(durations) / self._total_processing_time.total_seconds() if self._total_processing_time.total_seconds() > 0 else 0,
        }
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Calculate current data quality metrics.
        
        Returns:
            Dict: Data quality metrics
        """
        total_cells = self._frame.shape[0] * self._frame.shape[1]
        missing_cells = self._frame.isnull().sum().sum()
        
        return {
            'total_rows': self._frame.shape[0],
            'total_columns': self._frame.shape[1],
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'completeness_score': (total_cells - missing_cells) / total_cells if total_cells > 0 else 0.0,
            'duplicate_rows': self._frame.duplicated().sum(),
            'numeric_columns': len(self._frame.select_dtypes(include=[np.number]).columns),
            'text_columns': len(self._frame.select_dtypes(include=['object']).columns),
        }
    
    def compare_to_original(self) -> Dict[str, Any]:
        """
        Compare current DataFrame to original state.
        
        Returns:
            Dict: Comparison metrics
        """
        return {
            'original_shape': self._original_frame.shape,
            'current_shape': self._frame.shape,
            'rows_added': self._frame.shape[0] - self._original_frame.shape[0],
            'rows_removed': self._original_frame.shape[0] - self._frame.shape[0],
            'columns_added': self._frame.shape[1] - self._original_frame.shape[1],
            'columns_removed': self._original_frame.shape[1] - self._frame.shape[1],
            'shape_changed': self._frame.shape != self._original_frame.shape,
        }
    
    @classmethod
    def get_processor_by_id(cls, processor_id: str) -> Optional['DataProcessor']:
        """
        Retrieve a processor instance by its UUID.
        
        Args:
            processor_id: The UUID of the processor
            
        Returns:
            DataProcessor instance or None if not found
        """
        return cls._registry.get(processor_id)
    
    @classmethod
    def get_all_processors(cls) -> List['DataProcessor']:
        """
        Get all registered processor instances.
        
        Returns:
            List of all DataProcessor instances
        """
        return list(cls._registry.values())
    
    @classmethod
    def get_processors_by_type(cls, processor_type: str) -> List['DataProcessor']:
        """
        Get all processors of a specific type.
        
        Args:
            processor_type: The class name of the processor type
            
        Returns:
            List of matching DataProcessor instances
        """
        return [
            proc for proc in cls._registry.values()
            if proc.__class__.__name__ == processor_type
        ]
    
    @classmethod
    def clear_registry(cls):
        """Clear the processor registry (useful for testing)."""
        cls._registry.clear()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}(processor_id='{self._processor_id[:8]}...', "
            f"name='{self._name}', operations={self._processing_count})"
        )
    
    def __str__(self) -> str:
        """User-friendly representation."""
        ops = f"{self._processing_count} operations" if self._processing_count != 1 else "1 operation"
        return f"{self._name} ({self.__class__.__name__}): {ops} performed"
