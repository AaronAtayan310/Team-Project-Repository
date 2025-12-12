"""
Crime Research Data Pipeline - Enhanced Specialized Data Sources

This module contains enhanced specialized data source classes that inherit from
DataSource and provide source-specific behavior with advanced features including
UUID tracking, temporal metadata, and comprehensive source management.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

from .proj4_data_source import DataSource
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime
import time


class APIDataSource(DataSource):
    """
    Enhanced concrete implementation for loading data from REST APIs, inheriting from DataSource to gain UUID-based 
    identification, Temporal tracking, Registry pattern, Load history, and Data lineage tracking.
    
    Demonstrates:
    - Rate limiting tracking
    - Request timeout management
    - Response status tracking
    - Authentication support (basic)
    - Connection testing
    """
    
    def __init__(self, url: str, params: Optional[Dict] = None, 
                 timeout: int = 10, name: Optional[str] = None):
        """
        Initialize an enhanced API data source.
        
        Args:
            url: API endpoint URL
            params: Optional query parameters
            timeout: Request timeout in seconds
            name: Optional human-readable name
        """
        super().__init__(name=name)
        
        self.url = url
        self.params = params or {}
        self.timeout = timeout
        
        # Update metadata with API-specific info
        self._source_metadata['type'] = 'api'
        self._source_metadata['url'] = url
        self._source_metadata['timeout'] = timeout
        
        # API-specific tracking
        self._rate_limit_info: Dict[str, Any] = {}
        self._last_status_code: Optional[int] = None
    
    def validate_source(self) -> bool:
        """
        Validate that the URL is formed correctly.
        
        Returns:
            bool: True if valid
        """
        is_valid = self.url.startswith(('http://', 'https://'))
        self._source_metadata['validated'] = is_valid
        self._source_metadata['validated_at'] = datetime.now().isoformat()
        return is_valid
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the API connection without loading full data.
        
        Returns:
            Dict: Connection test results
        """
        if not self.validate_source():
            return {
                'success': False,
                'error': 'Invalid URL format',
                'url': self.url
            }
        
        try:
            import requests
        except ImportError:
            return {
                'success': False,
                'error': "'requests' library is required for API data sources"
            }
        
        start_time = time.time()
        try:
            response = requests.head(self.url, timeout=self.timeout)
            duration = time.time() - start_time
            
            return {
                'success': True,
                'status_code': response.status_code,
                'response_time': duration,
                'headers': dict(response.headers),
                'url': self.url
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'response_time': duration,
                'url': self.url
            }
    
    def load(self) -> pd.DataFrame:
        """
        Load data from API endpoint with enhanced tracking.
        
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If URL is invalid
            ImportError: If requests library is not available
        """
        if not self.validate_source():
            raise ValueError(f"Invalid API URL: {self.url}")
        
        try:
            import requests
        except ImportError:
            raise ImportError("'requests' library is required for API data sources")
        
        # Track load timing
        start_time = time.time()
        
        try:
            # Make API request
            response = requests.get(self.url, params=self.params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("API response must be JSON list or dict")
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update API-specific metadata
            self._last_status_code = response.status_code
            self._source_metadata['last_status_code'] = response.status_code
            self._source_metadata['last_response_time'] = duration
            
            # Extract rate limit info if available
            if 'X-RateLimit-Remaining' in response.headers:
                self._rate_limit_info = {
                    'remaining': response.headers.get('X-RateLimit-Remaining'),
                    'limit': response.headers.get('X-RateLimit-Limit'),
                    'reset': response.headers.get('X-RateLimit-Reset'),
                    'checked_at': datetime.now().isoformat()
                }
                self._source_metadata['rate_limit'] = self._rate_limit_info
            
            # Track the load using base class method
            self._track_load(df, duration, success=True)
            
            return df
            
        except Exception as e:
            duration = time.time() - start_time
            # Track failed load
            empty_df = pd.DataFrame()
            self._track_load(empty_df, duration, success=False)
            raise
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get current rate limit information (if available).
        
        Returns:
            Dict: Rate limit information or empty dict if not available
        """
        return self._rate_limit_info.copy()
    
    def __str__(self) -> str:
        """User-friendly representation."""
        status = f"loaded {self._load_count} times" if self._load_count > 0 else "not yet loaded"
        return f"{self._name} (API: {self.url}): {status}"


class CSVDataSource(DataSource):
    """
    Enhanced concrete implementation for loading data from CSV files.
    
    Inherits from DataSource to gain:
    - UUID-based identification
    - Temporal tracking
    - Registry pattern
    - Load history
    - Data lineage tracking
    
    Adds CSV-specific features:
    - File information (size, modified date)
    - Parse options support
    - Encoding detection
    - File validation
    """
    
    def __init__(self, filepath: str, name: Optional[str] = None, **read_csv_kwargs):
        """
        Initialize an enhanced CSV data source.
        
        Args:
            filepath: Path to the CSV file
            name: Optional human-readable name
            **read_csv_kwargs: Additional arguments to pass to pd.read_csv()
        """
        super().__init__(name=name)
        
        self.filepath = filepath
        self.read_csv_kwargs = read_csv_kwargs
        
        # Update metadata with CSV-specific info
        self._source_metadata['type'] = 'csv'
        self._source_metadata['filepath'] = filepath
        
        # CSV-specific attributes
        self._file_info: Dict[str, Any] = {}
    
    def validate_source(self) -> bool:
        """
        Validate that the CSV file exists and is readable.
        
        Returns:
            bool: True if valid
        """
        path = Path(self.filepath)
        is_valid = path.exists() and path.suffix.lower() == '.csv'
        
        self._source_metadata['validated'] = is_valid
        self._source_metadata['validated_at'] = datetime.now().isoformat()
        
        if is_valid:
            # Collect file information
            self._file_info = {
                'size_bytes': path.stat().st_size,
                'modified_at': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                'absolute_path': str(path.absolute())
            }
            self._source_metadata['file_info'] = self._file_info
        
        return is_valid
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get detailed file information.
        
        Returns:
            Dict: File information including size, modified date, etc.
        """
        if not self._file_info:
            self.validate_source()
        return self._file_info.copy()
    
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file with enhanced tracking.
        
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not self.validate_source():
            raise FileNotFoundError(f"CSV file not found or invalid: {self.filepath}")
        
        # Track load timing
        start_time = time.time()
        
        try:
            # Load CSV with any additional kwargs
            df = pd.read_csv(self.filepath, **self.read_csv_kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update CSV-specific metadata
            self._source_metadata['last_encoding'] = self.read_csv_kwargs.get('encoding', 'utf-8')
            
            # Track the load using base class method
            self._track_load(df, duration, success=True)
            
            return df
            
        except Exception as e:
            duration = time.time() - start_time
            # Track failed load
            empty_df = pd.DataFrame()
            self._track_load(empty_df, duration, success=False)
            raise
    
    def load_with_options(self, **new_kwargs) -> pd.DataFrame:
        """
        Load CSV with different parsing options.
        
        Args:
            **new_kwargs: New arguments to pass to pd.read_csv()
            
        Returns:
            pd.DataFrame: Loaded data
        """
        # Temporarily update kwargs
        old_kwargs = self.read_csv_kwargs.copy()
        self.read_csv_kwargs.update(new_kwargs)
        
        try:
            df = self.load()
        finally:
            # Restore original kwargs
            self.read_csv_kwargs = old_kwargs
        
        return df
    
    def __str__(self) -> str:
        """User-friendly representation."""
        status = f"loaded {self._load_count} times" if self._load_count > 0 else "not yet loaded"
        path = Path(self.filepath).name
        return f"{self._name} (CSV: {path}): {status}"


class DatabaseDataSource(DataSource):
    """
    Enhanced concrete implementation for loading data from databases, inheriting from DataSource to gain UUID-based 
    identification, Temporal tracking, Registry pattern, Load history, and Data lineage tracking.
    
    Demonstrates:
    - Connection pooling support
    - Query testing without execution
    - Table information retrieval
    - Query performance tracking
    """
    
    def __init__(self, connection_string: str, query: str, name: Optional[str] = None):
        """
        Initialize an enhanced database data source.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            name: Optional human-readable name
        """
        super().__init__(name=name)
        
        self.connection_string = connection_string
        self.query = query
        
        # Update metadata with database-specific info
        self._source_metadata['type'] = 'database'
        self._source_metadata['query_length'] = len(query)
        
        # Database-specific tracking
        self._last_query_plan: Optional[str] = None
    
    def validate_source(self) -> bool:
        """
        Validate that connection string and query are provided.
        
        Returns:
            bool: True if valid
        """
        is_valid = bool(self.connection_string and self.query)
        self._source_metadata['validated'] = is_valid
        self._source_metadata['validated_at'] = datetime.now().isoformat()
        return is_valid
    
    def test_query(self) -> Dict[str, Any]:
        """
        Test the query without executing it (if database supports EXPLAIN).
        
        Returns:
            Dict: Query test results
        """
        if not self.validate_source():
            return {
                'success': False,
                'error': 'Invalid configuration'
            }
        
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            return {
                'success': False,
                'error': 'SQLAlchemy required for query testing'
            }
        
        try:
            engine = create_engine(self.connection_string)
            
            # Try to get query plan (works for many SQL databases)
            with engine.connect() as conn:
                # Attempt EXPLAIN (works for PostgreSQL, MySQL, SQLite)
                try:
                    explain_query = f"EXPLAIN {self.query}"
                    result = conn.execute(text(explain_query))
                    plan = [str(row) for row in result]
                    self._last_query_plan = "\n".join(plan)
                    
                    return {
                        'success': True,
                        'query_plan': self._last_query_plan,
                        'query_valid': True
                    }
                except:
                    # EXPLAIN not supported or query has issues
                    return {
                        'success': True,
                        'query_plan': 'Query plan not available',
                        'query_valid': 'unknown'
                    }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    def load(self) -> pd.DataFrame:
        """
        Load data from database with enhanced tracking.
        
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If source validation fails
            ImportError: If SQLAlchemy is not installed
            Exception: If database connection or query execution fails
        """
        if not self.validate_source():
            raise ValueError(
                "Invalid Database Configuration. "
                "Connection string and query must be provided."
            )
        
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for database connections. "
                "Install it with: pip install sqlalchemy"
            )
        
        # Track load timing
        start_time = time.time()
        
        try:
            # Create database engine
            engine = create_engine(self.connection_string)
            
            # Execute query and load into DataFrame
            df = pd.read_sql(self.query, engine)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update database-specific metadata
            self._source_metadata['last_query_time'] = duration
            
            # Track the load using base class method
            self._track_load(df, duration, success=True)
            
            # Close the connection
            engine.dispose()
            
            return df
            
        except Exception as e:
            duration = time.time() - start_time
            # Track failed load
            empty_df = pd.DataFrame()
            self._track_load(empty_df, duration, success=False)
            
            # Update metadata with error information
            self._source_metadata['last_error'] = str(e)
            self._source_metadata['last_error_type'] = type(e).__name__
            
            # Re-raise with more context
            raise Exception(
                f"Failed to load data from database. "
                f"Connection: {self.connection_string}, "
                f"Error: {str(e)}"
            ) from e
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict: Table information or error
        """
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError:
            return {'error': 'SQLAlchemy required'}
        
        try:
            engine = create_engine(self.connection_string)
            inspector = inspect(engine)
            
            if table_name not in inspector.get_table_names():
                return {'error': f"Table '{table_name}' not found"}
            
            columns = inspector.get_columns(table_name)
            
            return {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True)
                    }
                    for col in columns
                ],
                'column_count': len(columns)
            }
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    def __str__(self) -> str:
        """User-friendly representation."""
        status = f"loaded {self._load_count} times" if self._load_count > 0 else "not yet loaded"
        # Mask sensitive connection info
        conn_type = self.connection_string.split('://')[0] if '://' in self.connection_string else 'database'
        return f"{self._name} (DB: {conn_type}): {status}"
