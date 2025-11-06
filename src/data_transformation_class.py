from sklearn.preprocessing import StandardScaler
import pandas as pd

class dataTransformation:
    """
    This class allows for transforming dataframes into scaled or more feature complete versions.

    Attributes:
        frame (pd.DataFrame): A pandas dataframe containing the data we are working with, holding relevant data.
    """
    def __init__(self, frame):
        """
        Initialize the dataTransformation class.

        Args:
            frame: A pandas dataframe containing the data we are working with, holding relevant data.
        """
        self._frame = None
        self.frame = frame


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
            raise ValueError('Data to transform must be in DataFrame format, no other format is acceptable.')
        self._frame = val

    def scale_features(self, columns: list[str]):
        """
        Scales the features of the dataframe we are working with.

        Args:
            columns (list[str]): Columns to scale.
        """
        df = self.frame.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.frame = df

    def generate_features(self):
        """
        Generate new derived features based on the data we are working with, update the dataframe to include them.
        """
        df = self.frame.copy()
        if "value" in df.columns and "count" in df.columns:
            df["value_per_count"] = df["value"] / (df["count"] + 1e-9)
        self.frame = df

    def __str__(self):
        """
        Returns a string representation of the dataTransformation object (the current state of the data being transformed).

        Returns:
            str: A readable description of the object.
        """
        source = str(self.frame)
        print('Current state of data being transformed:')
        print('\n')
        print(source)
