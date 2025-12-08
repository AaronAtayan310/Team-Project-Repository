"""
Crime Research Data Pipeline - Core Data Analysis Class

This module defines the dataAnalysis class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: OOP Class Implementation (Project 2)
"""

import pandas as pd
from typing import Any, Dict, List, Iterator, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class dataAnalysis:
    """
    This class uses pandas dataframes holding data of interest and inegrates them with instance methods allowing analysis of the data.

    Attributes:
        frame (pd.DataFrame): A pandas dataframe containing the data we are working with.
        __described (pd.DataFrame): Description of the dataframe we are working with, a
        a dataframe showing mean, deviation, so forth for the different columns.
    """
    def __init__(self, frame):
        """
        Initialize the dataAnalysis class.

        Args:
            frame: A pandas dataframe containing the data we are working with, holding relevant data.
        """
        self._frame = None
        self.frame = frame
        self.__described = self.frame.describe()

    @property
    def frame(self):
        """
        Gets the dataframe we are working with.
        """
        return self._frame

    @frame.setter
    def frame(self, val: pd.DataFrame):
        """
        Sets the value of the dataframe we are working with.

        Args:
            val: The value we are trying to assign to the dataframe.

        Raises:
            ValueError: If val is anything other than a pandas dataframe.
        """
        if not(isinstance(val, pd.DataFrame)):
            raise ValueError('Data to analyze must be in DataFrame format, no other format is acceptable.')
        self._frame = val

    @property
    def described(self) -> pd.DataFrame:
        """
        Gets the description of the dataframe we are working with (a dataframe showing mean, deviation, so forth for the different columns).
        """
        return self.__described
        
    def run_regression(self, y: pd.Series) -> LinearRegression:
        """
        Fit a simple linear regression model.

        Args:
            y (pd.Series): Target variable.

        Returns:
            LinearRegression: Trained regression model.
        """
        model = LinearRegression()
        model.fit(self.frame, y)
        return model
        
    def evaluate_model(self, model: LinearRegression, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained regression model using Mean Squared Error (MSE).

        Args:
            model (LinearRegression): Trained model.
            y_test (pd.Series): Test targets.

        Returns:
            dict: Dictionary with evaluation metrics.
        """
        predictions = model.predict(self.frame)
        mse = mean_squared_error(y_test, predictions)
        return {"mse": mse}
        
    def calculate_missing_data(self) -> pd.Series:
        """
        Calculate the percentage of missing data in each column of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            pd.Series: A Series containing the percentage of missing data per column.

        Raises:
            TypeError: If 'df' is not a pandas DataFrame.
        """
        return (self.frame.isnull().sum() / len(self.frame)) * 100
        
    def compute_crime_rate_by_year(self, population_col: str = "population") -> pd.DataFrame:
        """
        Compute annual crime rates per 100,000 people.

        Args:
            population_col (str): Column name representing population data.

        Returns:
            pd.DataFrame: DataFrame with columns ['year', 'crime_count', 'crime_rate'].
        """
        df = self.frame.copy()
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
        return yearly_data
        
    def top_crime_types(self, n: int = 10) -> pd.DataFrame:
        """
        Identify the top N most frequent crime types.

        Args:
            n (int): The number of top crime types to return.

        Returns:
            pd.DataFrame: DataFrame with columns ['crime_type', 'count'].
        """
        if "crime_type" not in self.frame.columns:
            raise ValueError("The dataset must include a 'crime_type' column.")
        df = self.frame.copy()
        return (
            df["crime_type"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"index": "crime_type", "crime_type": "count"})
        )
      
    def find_high_crime_areas(self, area_col: str = "neighborhood") -> pd.DataFrame:
        """
        Identify the areas with the highest number of reported crimes.

        Args:
            area_col (str): The name of the column representing geographic areas.

        Returns:
            pd.DataFrame: DataFrame of areas sorted by descending crime count.
        """
        if area_col not in self.frame.columns:
            raise ValueError(f"'{area_col}' column not found in dataset.")
        df = self.frame.copy()
        area_stats = (
            df.groupby(area_col)
            .size()
            .reset_index(name="crime_count")
            .sort_values(by="crime_count", ascending=False)
        )
        return area_stats

    def __str__(self):
        """
        Returns a string representation of the dataAnalysis object (the source and description dataframes, in string form).

        Returns:
            str: A readable description of the object.
        """
        source = str(self.frame)
        description = str(self.described)
        print('Source of data:')
        print('\n')
        print(source)
        print('\n')
        print('Description of data:')
        print('\n')
        print(description)

    def __repr__(self):
        """
        Returns a developer-targeted string representation of the dataAnalysis object, summarizing key information on the releated dataframe.

        Returns:
            str: A development-useful representation helpful for tasks like debugging.
        """
        return f"dataAnalysis(frame_shape={self.frame.shape}, described_shape={self.described.shape})"

