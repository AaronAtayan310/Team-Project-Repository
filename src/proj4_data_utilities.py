"""
Crime Research Data Pipeline - Enhanced Data Storage Utilities

This module defines the enhanced FinalDataStorageUtils class with advanced features
including file versioning, metadata tracking, compression support, and
comprehensive storage management.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

import os
import pandas as pd
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import hashlib
import uuid
import shutil


class FinalDataStorageUtils:
    """
    Enhanced utility class for data pipeline storage operations.
    
    Demonstrates:
    - UUID-based identification
    - File versioning and history
    - Metadata tracking for all saved files
    - Compression support
    - Storage statistics and reporting
    - Data integrity verification (checksums)
    - Batch operations
    """
    
    def __init__(self, base_output_dir: Optional[str] = None, log_level: int = logging.INFO, enable_versioning: bool = True):
        """
        Initialize an enhanced FinalDataStorageUtils object.
        
        Args:
            base_output_dir: Base directory for output files
            log_level: Logging level
            enable_versioning: Whether to enable file versioning
        """
        # UUID identification
        self._storage_id = str(uuid.uuid4())
        self._created_at = datetime.now()
        
        # Directory management
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path.cwd()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self._enable_versioning = enable_versioning
        
        # Tracking
        self._save_history: List[Dict[str, Any]] = []
        self._file_registry: Dict[str, Dict[str, Any]] = {}  # Track all saved files
        
        # Statistics
        self._total_saves = 0
        self._successful_saves = 0
        self._failed_saves = 0
        self._total_bytes_written = 0
        
        # Setup logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
    
    @property
    def storage_id(self) -> str:
        """Get the unique identifier for this storage instance."""
        return self._storage_id
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    @staticmethod
    def _setup_logging(log_level: int) -> None:
        """
        Configure logging for the pipeline.
        
        Args:
            log_level: A number representing the level that logging should occur at
        """
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """
        Calculate MD5 checksum of a file for integrity verification.
        
        Args:
            filepath: Path to the file
            
        Returns:
            str: MD5 checksum
        """
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def _get_file_size(self, filepath: Path) -> int:
        """
        Get file size in bytes.
        
        Args:
            filepath: Path to the file
            
        Returns:
            int: File size in bytes
        """
        return filepath.stat().st_size if filepath.exists() else 0
    
    def _version_file(self, filepath: Path) -> Optional[Path]:
        """
        Create a versioned copy of an existing file.
        
        Args:
            filepath: Path to the file to version
            
        Returns:
            Optional[Path]: Path to the versioned file, or None if no existing file
        """
        if not filepath.exists() or not self._enable_versioning:
            return None
        
        # Create versions directory
        versions_dir = filepath.parent / ".versions"
        versions_dir.mkdir(exist_ok=True)
        
        # Create versioned filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{filepath.stem}_v{timestamp}{filepath.suffix}"
        version_path = versions_dir / version_name
        
        # Copy file to versions directory
        shutil.copy2(filepath, version_path)
        
        return version_path
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str, use_timestamp: bool = False, compress: bool = False, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> Path:
        """
        Save a DataFrame to a CSV file with enhanced tracking.
        
        Args:
            df: The DataFrame to save
            filepath: The filepath for the CSV
            use_timestamp: Whether to add timestamp to filename
            compress: Whether to compress the file (gzip)
            metadata: Optional metadata to store with the file
            **kwargs: Additional arguments for df.to_csv()
            
        Returns:
            Path: Path object representing the saved file
        """
        path = Path(filepath)
        
        # Generate timestamped filename if requested
        if use_timestamp:
            base_name = path.stem
            extension = path.suffix
            timestamped_name = self.generate_timestamped_filename(base_name, extension)
            path = path.parent / timestamped_name
        
        # Version existing file if enabled
        version_path = None
        if path.exists() and self._enable_versioning:
            version_path = self._version_file(path)
        
        # Prepare save record
        save_record = {
            'save_id': str(uuid.uuid4()),
            'filepath': str(path.absolute()),
            'operation': 'save_csv',
            'started_at': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        self._total_saves += 1
        
        try:
            # Create parent directory
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine compression
            compression = 'gzip' if compress else None
            if compress and not path.suffix.endswith('.gz'):
                path = Path(str(path) + '.gz')
            
            # Save DataFrame
            df.to_csv(path, index=False, compression=compression, **kwargs)
            
            # Calculate file info
            file_size = self._get_file_size(path)
            checksum = self._calculate_checksum(path)
            
            # Update save record
            save_record.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'success',
                'file_size_bytes': file_size,
                'checksum': checksum,
                'rows_saved': len(df),
                'columns_saved': len(df.columns),
                'compressed': compress,
                'versioned_from': str(version_path) if version_path else None,
            })
            
            # Update file registry
            self._file_registry[str(path)] = {
                'filepath': str(path.absolute()),
                'file_type': 'csv',
                'created_at': datetime.now().isoformat(),
                'file_size_bytes': file_size,
                'checksum': checksum,
                'metadata': metadata or {}
            }
            
            self._successful_saves += 1
            self._total_bytes_written += file_size
            
            self.logger.info(f"CSV saved to: {path} ({file_size} bytes)")
            
        except Exception as e:
            # Update save record with error
            save_record.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            self._failed_saves += 1
            self.logger.error(f"Failed to save CSV: {e}")
            raise
        
        finally:
            # Always record the save attempt
            self._save_history.append(save_record)
        
        return path
    
    def load_from_csv(self, filepath: str, verify_checksum: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load a DataFrame from a CSV file with optional integrity verification.
        
        Args:
            filepath: The file path to load
            verify_checksum: Whether to verify file integrity using stored checksum
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            pd.DataFrame: The loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If checksum verification fails
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        # Verify checksum if requested and available
        if verify_checksum and str(path) in self._file_registry:
            stored_checksum = self._file_registry[str(path)].get('checksum')
            if stored_checksum:
                current_checksum = self._calculate_checksum(path)
                if current_checksum != stored_checksum:
                    raise ValueError(
                        f"Checksum verification failed for {path}. "
                        f"File may have been corrupted or modified."
                    )
        
        # Load DataFrame
        df = pd.read_csv(path, **kwargs)
        self.logger.info(f"CSV loaded from: {path} (shape: {df.shape})")
        
        return df
    
    def serialize_model(self, model: Any, path: str, metadata: Optional[Dict] = None, compress: bool = False) -> Path:
        """
        Serialize (save) a model object to disk with enhanced tracking.
        
        Args:
            model: The model to serialize
            path: The filepath for the serialized model
            metadata: Optional metadata about the model
            compress: Whether to compress the pickle file
            
        Returns:
            Path: Path object representing the saved file
        """
        path = Path(path)
        
        # Version existing file if enabled
        if path.exists() and self._enable_versioning:
            self._version_file(path)
        
        save_record = {
            'save_id': str(uuid.uuid4()),
            'filepath': str(path.absolute()),
            'operation': 'serialize_model',
            'started_at': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        self._total_saves += 1
        
        try:
            # Create parent directory
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize model
            with open(path, "wb") as file:
                pickle.dump(model, file)
            
            # Optionally compress
            if compress:
                import gzip
                with open(path, 'rb') as f_in:
                    with gzip.open(str(path) + '.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                path.unlink()  # Remove uncompressed version
                path = Path(str(path) + '.gz')
            
            # Calculate file info
            file_size = self._get_file_size(path)
            checksum = self._calculate_checksum(path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                self.logger.info(f"Model metadata saved to: {metadata_path}")
            
            # Update save record
            save_record.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'success',
                'file_size_bytes': file_size,
                'checksum': checksum,
                'compressed': compress,
            })
            
            # Update file registry
            self._file_registry[str(path)] = {
                'filepath': str(path.absolute()),
                'file_type': 'model',
                'created_at': datetime.now().isoformat(),
                'file_size_bytes': file_size,
                'checksum': checksum,
                'metadata': metadata or {}
            }
            
            self._successful_saves += 1
            self._total_bytes_written += file_size
            
            self.logger.info(f"Model serialized to: {path} ({file_size} bytes)")
            
        except Exception as e:
            save_record.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            self._failed_saves += 1
            self.logger.error(f"Failed to serialize model: {e}")
            raise
        
        finally:
            self._save_history.append(save_record)
        
        return path
    
    def deserialize_model(self, path: str, verify_checksum: bool = False) -> Any:
        """
        Deserialize (load) a model object from disk with optional verification.
        
        Args:
            path: The file path to load
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Any: The deserialized model
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If checksum verification fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Verify checksum if requested
        if verify_checksum and str(path) in self._file_registry:
            stored_checksum = self._file_registry[str(path)].get('checksum')
            if stored_checksum:
                current_checksum = self._calculate_checksum(path)
                if current_checksum != stored_checksum:
                    raise ValueError(
                        f"Checksum verification failed for {path}. "
                        f"File may have been corrupted."
                    )
        
        # Deserialize model
        with open(path, "rb") as file:
            model = pickle.load(file)
        
        self.logger.info(f"Model deserialized from: {path}")
        return model
    
    @staticmethod
    def generate_timestamped_filename(base_name: str, extension: str = ".csv") -> str:
        """
        Generate a timestamped filename.
        
        Args:
            base_name: The basic name without extension
            extension: The file extension (default: ".csv")
            
        Returns:
            str: The timestamped filename
        """
        if not isinstance(base_name, str) or not isinstance(extension, str):
            raise TypeError("Base name and extension must be strings")
        
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{base_name}_{timestamp}{extension}"
    
    def save_multiple_dataframes(self, dataframes: Dict[str, pd.DataFrame], output_dir: Optional[str] = None, use_timestamp: bool = False) -> List[Path]:
        """
        Save multiple DataFrames at once.
        
        Args:
            dataframes: Dictionary mapping filenames to DataFrames
            output_dir: Optional output directory (uses base_output_dir if None)
            use_timestamp: Whether to add timestamps to filenames
            
        Returns:
            List[Path]: List of saved file paths
        """
        output_dir = Path(output_dir) if output_dir else self.base_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for filename, df in dataframes.items():
            filepath = output_dir / filename
            path = self.save_to_csv(df, str(filepath), use_timestamp=use_timestamp)
            saved_paths.append(path)
        
        self.logger.info(f"Saved {len(saved_paths)} DataFrames to {output_dir}")
        return saved_paths
    
    def list_saved_files(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all files in the registry.
        
        Args:
            file_type: Optional filter by file type ('csv', 'model', etc.)
            
        Returns:
            List[Dict]: List of file information
        """
        files = list(self._file_registry.values())
        
        if file_type:
            files = [f for f in files if f.get('file_type') == file_type]
        
        return files
    
    def get_file_versions(self, filepath: str) -> List[Path]:
        """
        Get all versions of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            List[Path]: List of version file paths
        """
        path = Path(filepath)
        versions_dir = path.parent / ".versions"
        
        if not versions_dir.exists():
            return []
        
        # Find all versions of this file
        base_pattern = f"{path.stem}_v*{path.suffix}"
        versions = list(versions_dir.glob(base_pattern))
        
        return sorted(versions, reverse=True)  # Most recent first
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        
        Returns:
            Dict: Storage statistics
        """
        return {
            'storage_id': self._storage_id,
            'created_at': self._created_at.isoformat(),
            'base_output_dir': str(self.base_output_dir.absolute()),
            'total_saves': self._total_saves,
            'successful_saves': self._successful_saves,
            'failed_saves': self._failed_saves,
            'success_rate': self._successful_saves / self._total_saves if self._total_saves > 0 else 0.0,
            'total_bytes_written': self._total_bytes_written,
            'total_mb_written': self._total_bytes_written / (1024 * 1024),
            'files_tracked': len(self._file_registry),
            'versioning_enabled': self._enable_versioning,
        }
    
    def get_save_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get save operation history.
        
        Args:
            limit: Optional limit on number of records to return
            
        Returns:
            List[Dict]: Save history records
        """
        if limit:
            return self._save_history[-limit:]
        return self._save_history.copy()
    
    def __str__(self) -> str:
        """User-friendly representation."""
        return (
            f"FinalDataStorageUtils (id={self._storage_id[:8]}, "
            f"saves={self._successful_saves}/{self._total_saves}, "
            f"files_tracked={len(self._file_registry)}, "
            f"mb_written={self._total_bytes_written / (1024 * 1024):.2f})"
        )
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"FinalDataStorageUtils(storage_id='{self._storage_id[:8]}...', "
            f"successful_saves={self._successful_saves}, "
            f"failed_saves={self._failed_saves})"
        )
