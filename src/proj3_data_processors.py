"""
Crime Research Data Pipeline - Concrete Implementations Of Data Processors

This module contains specialized data processing classes that inherit from 
AbstractDataProcessor and provide type-specific behavior.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

from .proj3_base_classes import AbstractDataProcessor
import pandas as pd
import numpy as np
from typing import Optional, List, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class NewDataAnalysis(AbstractDataProcessor):
    """ 
    Class for statistical analysis and machine learning on DataFrames, inheritng
    from AbstractDataProcessor and specializes in analytical operations.
    """

    def __init__(self, frame: pd.DataFrame, verbose: bool = False):
        """ 
        Initalize the NewDataAnalysis object.
        
        Args:
            frame (pd.DataFrame): The DataFrame to analyze
            verbose (bool): If True, print analysis information
        """
        super().__init__(frame, verbose)
        self.__described = self._frame.describe()
        self._models = {} #This will store trained models
        self._log_operation("NewDataAnalysis object initialized")

    @property
    def described(self) -> pd.DataFrame:
        """ 
        Get description of DataFrame.
        """
        return self.__described
    
    def process(self) -> 'NewDataAnalysis':
        """ 
        Perform default analysis: update description stats.
        
        Returns:
            NewDataAnalysis: Self for method chaining
        """
        self._log_operation("Starting default analysis process")
        self.__described = self._frame.describe()

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
        Fit a simple linear regression model.
        
        Args:
            y (pd.Series): Target variable
            model_name (str): Name to store the model under
        
        Returns:
            LinearRegression: Trained regression model
        """
        model = LinearRegression()
        model.fit(self._frame. y)
        self._models[model_name] = model
        self._log_operation(f"Trained regression model '{model_name}'")
        
        return model
    
    def evaluate_model(self, model: LinearRegression, y_test: pd.Series) -> Dict[str, float]:
        """ 
        Evaluate a trained regression model using Mean Squared Error.
        
        Args:
            model (LinearRegression): Trained model
            y_test (pd.Series): Test targets
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        predictions = model.predict(self._frame)
        mse = mean_squared_error(y_test, predictions)
        self._log_operation(f"Model evaluated: MSE = {mse:.4f}")

        return {"mse": mse}

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
            population_col (str): Column name representing population data
            
        Returns:
            pd.DataFrame: DataFrame with columns ['year', 'crime_count', 'crime_rate']
        """
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
        
        self._log_operation("Computed crime rates by year")
        return yearly_data
    
    def top_crime_types(self, n: int = 10) -> pd.DataFrame:
        """ 
        Identify the top N most frequeny crime types.

        Args:
            n (int): The number of top crime types to return
            
        Returns:
            pd.DataFrame: DataFrame with column ['crime_type', 'count']
        """
        if "crime_type" not in self._frame.columns:
            raise ValueError("The dataset must include a 'crime_type' column")
        
        result = (
            self._frame["crime_type"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"crime_type": "count", "index": "crime_type"})
        )

        self._log_operation(f"Identified top {n} crime types")

        return result
    
     def find_high_crime_areas(self, area_col: str = "neighborhood") -> pd.DataFrame:
        """
        Identify the areas with the highest number of reported crimes.
        
        Args:
            area_col (str): The name of the column representing geographic areas
            
        Returns:
            pd.DataFrame: DataFrame of areas sorted by descending crime count
        """
        if area_col not in self._frame.columns:
            raise ValueError(f"'{area_col}' column not found in dataset.")
        
        area_stats = (
            self._frame.groupby(area_col)
            .size()
            .reset_index(name="crime_count")
            .sort_values(by="crime_count", ascending=False)
        )
        
        self._log_operation(f"Analyzed high crime areas by '{area_col}'")
        return area_stats
    
    def __str__(self) -> str:
        """
        Return a string representation of the NewDataAnalysis object.

        Returns:
            str: A readable, user-oriented string representation of the object.
        """
        lines = [
            "NewDataAnalysis (inherits from AbstractDataProcessor)",
            "=" * 60,
            f"DataFrame Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Trained Models: {len(self._models)}",
            f"Operations Performed: {len(self.processing_history)}",
            "=" * 60,
            "\nDescriptive Statistics:",
            str(self.__described)
        ]
        
        return "\n".join(lines)


class NewDataCleaner(AbstractDataProcessor):
    """
    Class for cleaning and preprocessing Pandas dataframes that 
    provides methods for missing values, standardizing column names, 
    normalizing text, removing outliers, and other common tasks.
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = False):
        """
        Initialize the data cleaner with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
            verbose (bool): If True, print information about cleaning operations
            
        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If df is empty 
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        super().__init__(df, verbose)
        self._original_shape = df.shape
        self._log_operation("NewDataCleaner object Initalized")
    
    @property
    def df(self) -> pd.DataFrame:
        """
        Get the current DataFrame.
        """
        return self._df
    
    @df.setter
    def df(self, value: pd.DataFrame):
        """
        Set the DataFrame with validation.

        Args:
            value (pd.DataFrame): What the dataframe, df, should look like after setter usage.
        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Value must be a pandas DataFrame")
        self._df = value

    @property
    def original_shape(self) -> tuple:
        """
        Get the original shape of the DataFrame.
        """
        return self._original_shape
    
    @property
    def cleaning_history(self) -> List[str]:
        """
        Get the history of cleaning operations.
        """
        return self.processing_history
        
    @property
    def verbose(self) -> bool:
        """
        Get verbose setting.
        """
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: bool):
        """
        Set verbose setting.

        Args:
            value (bool): What the setting should match after setter usage.
        """
        if not isinstance(value, bool):
            raise TypeError("Verbose must be a boolean")
        self._verbose = value

    def _log_operation(self, operation: str):
        """
        Log a cleaning operation to the history.

        Args:
            operation (str): Operation to be logged.
        """
        self._cleaning_history.append(operation)
        if self._verbose:
            print(f"[NewDataCleaner] {operation}")

    def handle_missing_values(self, strategy: str = "mean", columns: Optional[List[str]] = None) -> 'NewDataCleaner':
        """
        Handle missing values in the DataFrame using a given strategy.

        Args:
            strategy (str): Method to handle missing values
            columns (Optional[List[str]]): Specific columns to apply strategy to. If None, applies to all columns
            
        Returns:
            NewDataCleaner: Self for method chaining
            
        Raises:
            ValueError: If strategy is invalid
        """
        valid_strategies = ['mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid Strategy. Choose from {valid_strategies}")
        
        target_df = self._df[columns] if columns else self._df
        missing_before = self._df.isnull().sum().sum()

        if strategy == "mean":
            if columns:
                self._df[columns] = self._df[columns].fillna(self._df[columns].mean(numeric_only=True))
            else:
                self._df.fillna(self._df.mean(numeric_only=True), inplace=True)
        
        elif strategy == "median":
            if columns:
                self._df[columns] = self._df[columns].fillna(self._df[columns].median(numeric_only=True))
            else:
                self._df.fillna(self._df.median(numeric_only=True), inplace=True)
        
        elif strategy == "mode":
            if columns:
                for col in columns:
                    if not self._df[col].mode().empty:
                        self._df[col].fillna(self._df[col].mode()[0], inplace=True)
            else:
                for col in self._df.columns:
                    if not self._df[col].mode().empty:
                        self._df[col].fillna(self._df[col].mode()[0], inplace=True)
        
        elif strategy == "drop":
            self._df.dropna(subset=columns, inplace=True)
        
        elif strategy == "forward_fill":
            if columns:
                self._df[columns] = self._df[columns].fillna(method='ffill')
            else:
                self._df.fillna(method='ffill', inplace=True)
        
        elif strategy == "backward_fill":
            if columns:
                self._df[columns] = self._df[columns].fillna(method='bfill')
            else:
                self._df.fillna(method='bfill', inplace=True)

        missing_after = self._df.isnull().sum().sum()
        cols_msg = f" in columns {columns}" if columns else ""
        self._log_operation(f"Handled missing values using '{strategy}' strategy{cols_msg}")
        return self
    
    def normalize_text_column(self, column: str, remove_special_chars: bool = False) -> 'NewDataCleaner':
        """
        Normalize the text in a specified column.

        Args:
            column (str): Column to normalize
            remove_special_chars (bool): If True, remove special characters
            
        Returns:
            NewDataCleaner: Self for method chaining
            
        Raises:
            ValueError: If column doesn't exist in DataFrame
        """
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        self._df[column] = self._df[column].astype(str).str.lower().str.strip()

        if remove_special_chars:
            self._df[column] = self._df[column].str.replace(r'[^a-z0-9\s]', '', regex = True)

        self._log_operation(f"Normalized text column '{column}'")
        return self
    
    def __str__(self) -> str:
        """
        Returns a string representation of the NewDataCleaner object.

        Returns:
            str: Formatted summary
        """

        missing_values = self._df.isnull().sum().sum()
        missing_pct = (
            missing_values / (self._df.shape[0] * self._df.shape[1])) * 100 if self._df.size > 0 else 0
        
        lines = [
            "NewDataCleaner Summary",
            "=" * 50,
            f"Current Shape: {self._df.shape[0]} rows × {self._df.shape[1]} columns",
            f"Original Shape: {self._original_shape[0]} rows × {self._original_shape[1]} columns",
            f"Missing Values: {missing_values} ({missing_pct:.2f}%)",
            f"Operations Performed: {len(self._cleaning_history)}",
            "=" * 50
        ]

        if self._cleaning_history:
            lines.append("Cleaning History:")
            for i, operation in enumerate(self._cleaning_history, 1):
                lines.append(f"  {i}. {operation}")
        else:
            lines.append("No cleaning operations performed")

        return "\n".join(lines)
    
    def process(self) -> 'NewDataCleaner':
        """
        Perform default cleaning process: handle missing values with mean strat.
        
        Returns:
            NewDataCleaner: Self for method chaining
        """
        self._log_operation("Starting default cleaning process")
        self.handle_missing_values(strategy = 'mean')
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


class NewDataTransformation(AbstractDataProcessor):
    """
    Class for transforming DataFrames through scaling and feature engineering, inheriting
    from AbstractDataProcessor and specializes in feature transformation operations
    """

    def __init__(self, frame: pd.DataFrame, verbose: bool = False):
        """ 
        Initialize the NewDataTransformation object.
        
        Args:
            frame (pd.DataFrame): The DataFrame to transform
            verbose (bool): If True, print transformation information
        """
        super().__init__(frame, verbose)
        self.scalers = {}
        self._log_operastion("NewDataTransformation object initialized")

    def process(self) -> 'NewDataTransformation':
        """
        Perform default transformation: generate features.
        
        Returns:
            NewDataTransformation: Self for method chaining
        """
        self._log_operation("Starting default process")
        self.generate_features()
        
        return self
    
    def validate(self) -> bool:
        """ 
        Validates that the DataFrame has no infinite values and appropriate dtypes.
        
        Returns:
            bool: True if DataFrame is valid for transformation
        """
        has_no_inf = not np.isinf(self._frame.select_dtypes(include=[np.number])).any().any()
        has_numeric = len(self.frame.select_dtypes(include=[np.number]).columns) > 0
        is_valid = has_no_inf and has_numeric

        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_no_inf:
                print("  - Infinite values detected")
            if not has_numeric:
                print("  - No numeric columns for transformation")
        
        return is_valid
    
    def scale_features(self, columns: List[str]) -> 'NewDataTransformation':
        """ 
        Scale the features of the DataFrame using StandardScaler.
        
        Args:
            columns(List[str]): Columns to scale
            
        Returns:
            NewDataTransformation: Self for method chaining
        """
        df = self.frame.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])

        #Storing scaler for future use:
        self._scalers[tuple(columns)] = scaler

        self._frame = df
        self._log_operations(f"Scaled features: {columns}")

        return self
    
    def generate_features(self) -> 'NewDataTransformation':
        """ 
        Generate new derived features based on existing data.
        
        Returns:
            NewDataTransformation: Self for method chaining
        """
        df = self._frame.copy()
        features_added = []

        if "value" in df.columns and "count" in df.columns:
            df["value_per_count"] = df["value"] / (df["count"] + 1e-9)
            features_added.append("value_per_count")
        
        self._frame = df

        if features_added:
            self._log_operation(f"Generated features: {features_added}")
        else:
            self._log_operation("No features generated (required columns not found)")
        
        return self
    
    def __str__(self) -> str:
        """
        Return a string representation of the NewDataTransformation object.
        """
        lines = [
            "NewDataTransformation (inherits from AbstractDataProcessor)",
            "=" * 60,
            f"Current Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Scaled Feature Sets: {len(self._scalers)}",
            f"Operations Performed: {len(self.processing_history)}",
            "=" * 60
        ]
        
        if self.processing_history:
            lines.append("Processing History:")
            for operation in self.processing_history[-5:]:
                lines.append(f"  • {operation}")
        
        return "\n".join(lines)
