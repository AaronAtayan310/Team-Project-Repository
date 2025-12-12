"""
Crime Research Data Pipeline - Enhanced Specialized Data Processors

This module contains enhanced specialized data processing classes that inherit
from DataProcessor and provide type-specific behavior with advanced features
including UUID tracking, performance monitoring, and data quality validation.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

from .proj4_data_processor import DataProcessor
from .proj4_data_quality_standards import DataQualityStandards, QualityLevel
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time


class CrimeDataAnalysis(DataProcessor):
    """
    Enhanced class for statistical analysis and machine learning on crime DataFrames. inheriting from DataProcessor to gain UUID-based identification, 
    temporal tracking with operation durations, registry pattern, performance monitoring, and data quality tracking.
    
    Demonstrates:
    - Model metadata tracking
    - Quality score integration
    - Crime-specific analytics (preserved from P3)
    - Enhanced result dictionaries with metadata
    """
    
    def __init__(self, frame: pd.DataFrame, verbose: bool = False, name: Optional[str] = None, quality_standards: Optional[DataQualityStandards] = None):
        """
        Initialize the CrimeDataAnalysis object.
        
        Args:
            frame: The DataFrame to analyze
            verbose: If True, print analysis information
            name: Optional human-readable name
            quality_standards: Optional DataQualityStandards for validation
        """
        super().__init__(frame, verbose, name)
        
        # Analysis-specific attributes
        self.__described = self._frame.describe()
        self._models: Dict[str, Dict[str, Any]] = {}  # Store models with metadata
        self._quality_standards = quality_standards
        
        # Log initialization with timing
        self._log_operation("CrimeDataAnalysis initialized", duration=0.0)
    
    @property
    def described(self) -> pd.DataFrame:
        """Get description of DataFrame."""
        return self.__described
    
    @property
    def models(self) -> Dict[str, Dict[str, Any]]:
        """Get all trained models with metadata."""
        return {k: v.copy() for k, v in self._models.items()}
    
    def process(self) -> 'CrimeDataAnalysis':
        """
        Perform default analysis: update description stats.
        
        Returns:
            CrimeDataAnalysis: Self for method chaining
        """
        start_time = time.time()
        
        self.__described = self._frame.describe()
        
        duration = time.time() - start_time
        self._log_operation("Processed: Updated descriptive statistics", duration=duration)
        
        # Increment processing count
        self._processing_count += 1
        
        return self
    
    def validate(self) -> bool:
        """
        Validate that the DataFrame has sufficient data for analysis.
        
        Returns:
            bool: True if DataFrame has enough rows and numeric columns
        """
        has_enough_rows = len(self._frame) >= 2
        has_numeric = len(self._frame.select_dtypes(include=[np.number]).columns) > 0
        is_valid = has_enough_rows and has_numeric
        
        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_enough_rows:
                print(f"  - Insufficient rows: {len(self._frame)} (need at least 2)")
            if not has_numeric:
                print("  - No numeric columns for analysis")
        
        return is_valid
    
    def run_regression(self, y: pd.Series, model_name: str = "default") -> LinearRegression:
        """
        Fit a simple linear regression model with enhanced metadata tracking.
        
        Args:
            y: Target variable
            model_name: Name to store the model under
            
        Returns:
            LinearRegression: Trained regression model
        """
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(self._frame, y)
        
        duration = time.time() - start_time
        
        # Store model with enhanced metadata
        self._models[model_name] = {
            'model': model,
            'trained_at': pd.Timestamp.now().isoformat(),
            'training_duration': duration,
            'feature_count': self._frame.shape[1],
            'sample_count': self._frame.shape[0],
            'coefficients': model.coef_.tolist() if hasattr(model.coef_, 'tolist') else None,
            'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None,
        }
        
        self._log_operation(
            f"Trained regression model '{model_name}'",
            duration=duration,
            metadata={'model_name': model_name, 'features': self._frame.shape[1]}
        )
        
        return model
    
    def evaluate_model(self, model: LinearRegression, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a trained regression model with enhanced metrics.
        
        Args:
            model: Trained model
            y_test: Test targets
            
        Returns:
            Dict: Enhanced evaluation metrics with metadata
        """
        start_time = time.time()
        
        predictions = model.predict(self._frame)
        mse = mean_squared_error(y_test, predictions)
        
        duration = time.time() - start_time
        
        result = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'evaluated_at': pd.Timestamp.now().isoformat(),
            'evaluation_duration': duration,
            'sample_count': len(y_test),
        }
        
        self._log_operation(
            f"Model evaluated: MSE = {mse:.4f}",
            duration=duration,
            metadata=result
        )
        
        return result
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all trained models with their metadata.
        
        Returns:
            List[Dict]: Model information
        """
        return [
            {
                'name': name,
                'trained_at': info['trained_at'],
                'feature_count': info['feature_count'],
                'sample_count': info['sample_count'],
            }
            for name, info in self._models.items()
        ]
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model metadata or None if not found
        """
        if model_name not in self._models:
            return None
        
        info = self._models[model_name].copy()
        # Remove the actual model object from the returned info
        info.pop('model', None)
        return info
    
    def calculate_missing_data(self) -> pd.Series:
        """
        Calculate the percentage of missing data in each column.
        
        Returns:
            pd.Series: A series containing the percentage of missing data
        """
        return (self._frame.isnull().sum() / len(self._frame)) * 100
    
    def compute_crime_rate_by_year(self, population_col: str = "population") -> pd.DataFrame:
        """
        Compute annual crime rates per 100,000 people.
        
        Args:
            population_col: Column name representing population data
            
        Returns:
            pd.DataFrame: DataFrame with columns ['year', 'crime_count', 'crime_rate']
        """
        start_time = time.time()
        
        df = self._frame.copy()
        if "date" not in df.columns:
            raise ValueError("The dataset must contain a 'date' column.")
        if population_col not in df.columns:
            raise ValueError(f"Missing '{population_col}' column for population data.")
        
        df["year"] = pd.to_datetime(df["date"]).dt.year
        yearly_data = df.groupby("year").agg(
            crime_count=("crime_type", "count"),
            population=(population_col, "mean")
        ).reset_index()
        yearly_data["crime_rate"] = (yearly_data["crime_count"] / yearly_data["population"]) * 100000
        
        duration = time.time() - start_time
        self._log_operation("Computed crime rates by year", duration=duration)
        
        return yearly_data
    
    def top_crime_types(self, n: int = 10) -> pd.DataFrame:
        """
        Identify the top N most frequent crime types.
        
        Args:
            n: The number of top crime types to return
            
        Returns:
            pd.DataFrame: DataFrame with columns ['crime_type', 'count']
        """
        start_time = time.time()
        
        if "crime_type" not in self._frame.columns:
            raise ValueError("The dataset must include a 'crime_type' column")
        
        result = (
            self._frame["crime_type"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"index": "crime_type", "count": "count"})
        )
        
        # Ensure columns are named correctly regardless of pandas version
        if 'crime_type' not in result.columns:
            result.columns = ['crime_type', 'count']
        
        duration = time.time() - start_time
        self._log_operation(f"Identified top {n} crime types", duration=duration)
        
        return result
    
    def find_high_crime_areas(self, area_col: str = "neighborhood") -> pd.DataFrame:
        """
        Identify the areas with the highest number of reported crimes.
        
        Args:
            area_col: The name of the column representing geographic areas
            
        Returns:
            pd.DataFrame: DataFrame of areas sorted by descending crime count
        """
        start_time = time.time()
        
        if area_col not in self._frame.columns:
            raise ValueError(f"'{area_col}' column not found in dataset.")
        
        area_stats = (
            self._frame.groupby(area_col)
            .size()
            .reset_index(name="crime_count")
            .sort_values(by="crime_count", ascending=False)
        )
        
        duration = time.time() - start_time
        self._log_operation(f"Analyzed high crime areas by '{area_col}'", duration=duration)
        
        return area_stats
    
    def get_analysis_with_quality(self) -> Dict[str, Any]:
        """
        Get analysis results with integrated quality assessment.
        
        Returns:
            Dict: Analysis summary with quality scores
        """
        result = {
            'summary': self.get_summary(),
            'performance': self.get_performance_metrics(),
            'data_quality': self.get_data_quality_metrics(),
            'model_count': len(self._models),
        }
        
        # Add quality standards assessment if available
        if self._quality_standards:
            quality_score = self._quality_standards.calculate_quality_score(self._frame)
            result['quality_assessment'] = quality_score
        
        return result
    
    def __str__(self) -> str:
        """User-friendly representation."""
        lines = [
            f"{self._name} (CrimeDataAnalysis)",
            "=" * 60,
            f"DataFrame Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Trained Models: {len(self._models)}",
            f"Operations Performed: {len(self.processing_history)}",
            f"Total Processing Time: {self._total_processing_time.total_seconds():.2f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


class CrimeDataCleaner(DataProcessor):
    """
    Enhanced class for cleaning and preprocessing on crime DataFrames, inheriting from DataProcessor to gain UUID-based identification, 
    temporal tracking with operation durations, registry pattern, performance monitoring, and data quality tracking.
    
    Demonstrates:
    - Quality validation integration
    - Before/after quality comparison
    - Cleaning impact metrics
    """
    
    def __init__(self, df: pd.DataFrame, verbose: bool = False, name: Optional[str] = None, quality_standards: Optional[DataQualityStandards] = None):
        """
        Initialize the CrimeDataCleaner.
        
        Args:
            df: The DataFrame to clean
            verbose: If True, print information about cleaning operations
            name: Optional human-readable name
            quality_standards: Optional DataQualityStandards for validation
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        super().__init__(df, verbose, name)
        
        # Cleaning-specific attributes (original_shape inherited from base as _original_frame)
        self._quality_standards = quality_standards
        self._quality_before: Optional[Dict[str, Any]] = None
        self._quality_after: Optional[Dict[str, Any]] = None
        
        # Calculate initial quality if standards provided
        if self._quality_standards:
            self._quality_before = self._quality_standards.calculate_quality_score(self._frame)
        
        self._log_operation("CrimeDataCleaner initialized", duration=0.0)
    
    @property
    def cleaning_history(self) -> List[Dict[str, Any]]:
        """Get the history of cleaning operations."""
        return self.processing_history
    
    def handle_missing_values(self, strategy: str = "mean", columns: Optional[List[str]] = None) -> 'CrimeDataCleaner':
        """
        Handle missing values with enhanced tracking.
        
        Args:
            strategy: Method to handle missing values
            columns: Specific columns to apply strategy to
            
        Returns:
            CrimeDataCleaner: Self for method chaining
        """
        start_time = time.time()
        
        valid_strategies = ['mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid Strategy. Choose from {valid_strategies}")
        
        missing_before = self._frame.isnull().sum().sum()
        
        if strategy == "mean":
            if columns:
                self._frame[columns] = self._frame[columns].fillna(
                    self._frame[columns].mean(numeric_only=True)
                )
            else:
                self._frame.fillna(self._frame.mean(numeric_only=True), inplace=True)
        
        elif strategy == "median":
            if columns:
                self._frame[columns] = self._frame[columns].fillna(
                    self._frame[columns].median(numeric_only=True)
                )
            else:
                self._frame.fillna(self._frame.median(numeric_only=True), inplace=True)
        
        elif strategy == "mode":
            if columns:
                for col in columns:
                    if not self._frame[col].mode().empty:
                        self._frame[col].fillna(self._frame[col].mode()[0], inplace=True)
            else:
                for col in self._frame.columns:
                    if not self._frame[col].mode().empty:
                        self._frame[col].fillna(self._frame[col].mode()[0], inplace=True)
        
        elif strategy == "drop":
            self._frame.dropna(subset=columns, inplace=True)
        
        elif strategy == "forward_fill":
            if columns:
                self._frame[columns] = self._frame[columns].fillna(method='ffill')
            else:
                self._frame.fillna(method='ffill', inplace=True)
        
        elif strategy == "backward_fill":
            if columns:
                self._frame[columns] = self._frame[columns].fillna(method='bfill')
            else:
                self._frame.fillna(method='bfill', inplace=True)
        
        missing_after = self._frame.isnull().sum().sum()
        duration = time.time() - start_time
        
        cols_msg = f" in columns {columns}" if columns else ""
        self._log_operation(
            f"Handled missing values using '{strategy}' strategy{cols_msg}",
            duration=duration,
            metadata={
                'strategy': strategy,
                'missing_before': missing_before,
                'missing_after': missing_after,
                'cells_cleaned': missing_before - missing_after
            }
        )
        
        return self
    
    def normalize_text_column(self, column: str, remove_special_chars: bool = False) -> 'CrimeDataCleaner':
        """
        Normalize text in a column with enhanced tracking.
        
        Args:
            column: Column to normalize
            remove_special_chars: If True, remove special characters
            
        Returns:
            CrimeDataCleaner: Self for method chaining
        """
        start_time = time.time()
        
        if column not in self._frame.columns:
            raise ValueError(f"Column '{column}' not found")
        
        self._frame[column] = self._frame[column].astype(str).str.lower().str.strip()
        
        if remove_special_chars:
            self._frame[column] = self._frame[column].str.replace(r'[^a-z0-9\s]', '', regex=True)
        
        duration = time.time() - start_time
        self._log_operation(
            f"Normalized text column '{column}'",
            duration=duration,
            metadata={'column': column, 'removed_special_chars': remove_special_chars}
        )
        
        return self
    
    def process(self) -> 'CrimeDataCleaner':
        """
        Perform default cleaning process with quality tracking.
        
        Returns:
            CrimeDataCleaner: Self for method chaining
        """
        start_time = time.time()
        
        self.handle_missing_values(strategy='mean')
        
        # Calculate quality after cleaning if standards provided
        if self._quality_standards:
            self._quality_after = self._quality_standards.calculate_quality_score(self._frame)
        
        duration = time.time() - start_time
        self._log_operation("Completed default cleaning process", duration=duration)
        
        self._processing_count += 1
        
        return self
    
    def validate(self) -> bool:
        """
        Validate that the DataFrame is in a clean state.
        
        Returns:
            bool: True if no missing values and no duplicate rows
        """
        has_no_missing = self._frame.isnull().sum().sum() == 0
        has_no_duplicates = not self._frame.duplicated().any()
        is_valid = has_no_missing and has_no_duplicates
        
        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_no_missing:
                print(f"  - Missing values detected: {self._frame.isnull().sum().sum()}")
            if not has_no_duplicates:
                print(f"  - Duplicate rows detected: {self._frame.duplicated().sum()}")
        
        return is_valid
    
    def get_cleaning_impact(self) -> Dict[str, Any]:
        """
        Get metrics showing the impact of cleaning operations.
        
        Returns:
            Dict: Cleaning impact metrics
        """
        comparison = self.compare_to_original()
        
        result = {
            'rows_removed': max(0, -comparison['rows_added']),
            'rows_added': max(0, comparison['rows_added']),
            'shape_changed': comparison['shape_changed'],
            'original_missing': self._original_frame.isnull().sum().sum(),
            'current_missing': self._frame.isnull().sum().sum(),
            'missing_resolved': self._original_frame.isnull().sum().sum() - self._frame.isnull().sum().sum(),
        }
        
        # Add quality comparison if available
        if self._quality_before and self._quality_after:
            result['quality_before'] = self._quality_before['overall_score']
            result['quality_after'] = self._quality_after['overall_score']
            result['quality_improvement'] = (
                self._quality_after['overall_score'] - self._quality_before['overall_score']
            )
        
        return result
    
    def __str__(self) -> str:
        """User-friendly representation."""
        missing_values = self._frame.isnull().sum().sum()
        missing_pct = (
            missing_values / (self._frame.shape[0] * self._frame.shape[1]) * 100 
            if self._frame.size > 0 else 0
        )
        
        lines = [
            f"{self._name} (CrimeDataCleaner)",
            "=" * 50,
            f"Current Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Original Shape: {self._original_frame.shape[0]} rows × {self._original_frame.shape[1]} columns",
            f"Missing Values: {missing_values} ({missing_pct:.2f}%)",
            f"Operations Performed: {len(self._processing_history)}",
            f"Total Processing Time: {self._total_processing_time.total_seconds():.2f}s",
            "=" * 50
        ]
        
        return "\n".join(lines)


class CrimeDataTransformation(DataProcessor):
    """
    Enhanced class for transforming crime DataFrames through scaling and feature engineering, inheriting from DataProcessor to gain UUID-based 
    identification, temporal tracking with operation durations, registry pattern, performance monitoring, and data quality tracking.
    
    Demonstrates:
    - Feature lineage tracking
    - Scaler metadata storage
    - Transformation impact metrics
    """
    
    def __init__(self, frame: pd.DataFrame, verbose: bool = False, name: Optional[str] = None):
        """
        Initialize the CrimeDataTransformation object.
        
        Args:
            frame: The DataFrame to transform
            verbose: If True, print transformation information
            name: Optional human-readable name
        """
        super().__init__(frame, verbose, name)
        
        # Transformation-specific attributes
        self._scalers: Dict[tuple, Dict[str, Any]] = {}  # Store scalers with metadata
        self._feature_lineage: List[Dict[str, Any]] = []  # Track created features
        
        self._log_operation("CrimeDataTransformation initialized", duration=0.0)
    
    @property
    def scalers(self) -> Dict[tuple, Dict[str, Any]]:
        """Get all fitted scalers with metadata."""
        return {k: v.copy() for k, v in self._scalers.items()}
    
    @property
    def feature_lineage(self) -> List[Dict[str, Any]]:
        """Get the lineage of created features."""
        return [f.copy() for f in self._feature_lineage]
    
    def process(self) -> 'CrimeDataTransformation':
        """
        Perform default transformation: generate features.
        
        Returns:
            CrimeDataTransformation: Self for method chaining
        """
        start_time = time.time()
        
        self.generate_features()
        
        duration = time.time() - start_time
        self._log_operation("Completed default transformation process", duration=duration)
        
        self._processing_count += 1
        
        return self
    
    def validate(self) -> bool:
        """
        Validate that the DataFrame has no infinite values and appropriate dtypes.
        
        Returns:
            bool: True if DataFrame is valid for transformation
        """
        has_no_inf = not np.isinf(
            self._frame.select_dtypes(include=[np.number])
        ).any().any()
        has_numeric = len(self.frame.select_dtypes(include=[np.number]).columns) > 0
        is_valid = has_no_inf and has_numeric
        
        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_no_inf:
                print("  - Infinite values detected")
            if not has_numeric:
                print("  - No numeric columns for transformation")
        
        return is_valid
    
    def scale_features(self, columns: List[str]) -> 'CrimeDataTransformation':
        """
        Scale features with enhanced metadata tracking.
        
        Args:
            columns: Columns to scale
            
        Returns:
            CrimeDataTransformation: Self for method chaining
        """
        start_time = time.time()
        
        df = self.frame.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        
        # Store scaler with enhanced metadata
        self._scalers[tuple(columns)] = {
            'scaler': scaler,
            'fitted_at': pd.Timestamp.now().isoformat(),
            'columns': columns,
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
        }
        
        self._frame = df
        
        duration = time.time() - start_time
        self._log_operation(
            f"Scaled features: {columns}",
            duration=duration,
            metadata={'columns': columns, 'method': 'StandardScaler'}
        )
        
        return self
    
    def generate_features(self) -> 'CrimeDataTransformation':
        """
        Generate new derived features with lineage tracking.
        
        Returns:
            CrimeDataTransformation: Self for method chaining
        """
        start_time = time.time()
        
        df = self._frame.copy()
        features_added = []
        
        if "value" in df.columns and "count" in df.columns:
            df["value_per_count"] = df["value"] / (df["count"] + 1e-9)
            features_added.append("value_per_count")
            
            # Track feature lineage
            self._feature_lineage.append({
                'feature_name': 'value_per_count',
                'created_at': pd.Timestamp.now().isoformat(),
                'derived_from': ['value', 'count'],
                'formula': 'value / (count + 1e-9)'
            })
        
        self._frame = df
        
        duration = time.time() - start_time
        
        if features_added:
            self._log_operation(
                f"Generated features: {features_added}",
                duration=duration,
                metadata={'features': features_added}
            )
        else:
            self._log_operation(
                "No features generated (required columns not found)",
                duration=duration
            )
        
        return self
    
    def get_feature_lineage(self) -> List[Dict[str, Any]]:
        """
        Get the lineage of all created features.
        
        Returns:
            List[Dict]: Feature lineage information
        """
        return self._feature_lineage.copy()
    
    def inverse_transform(self, columns: List[str]) -> 'CrimeDataTransformation':
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            columns: Columns to inverse transform
            
        Returns:
            CrimeDataTransformation: Self for method chaining
        """
        start_time = time.time()
        
        key = tuple(columns)
        if key not in self._scalers:
            raise ValueError(f"No scaler found for columns: {columns}")
        
        scaler_info = self._scalers[key]
        scaler = scaler_info['scaler']
        
        df = self._frame.copy()
        df[columns] = scaler.inverse_transform(df[columns])
        self._frame = df
        
        duration = time.time() - start_time
        self._log_operation(
            f"Inverse transformed features: {columns}",
            duration=duration,
            metadata={'columns': columns}
        )
        
        return self
    
    def __str__(self) -> str:
        """User-friendly representation."""
        lines = [
            f"{self._name} (CrimeDataTransformation)",
            "=" * 60,
            f"Current Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Scaled Feature Sets: {len(self._scalers)}",
            f"Generated Features: {len(self._feature_lineage)}",
            f"Operations Performed: {len(self.processing_history)}",
            f"Total Processing Time: {self._total_processing_time.total_seconds():.2f}s",
            "=" * 60
        ]
        
        return "\n".join(lines)
