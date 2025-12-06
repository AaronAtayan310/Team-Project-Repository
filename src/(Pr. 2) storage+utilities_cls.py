# WIP

import pandas as pd
import pickle
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
import hashlib


class dataStorageUtils:
    """
    Utility class for data pipeline operations including storage, serialization,
    logging, and file management.
    """
    
    def __init__(self, base_output_dir: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize the dataStorageUtils class.
        
        Args:
            base_output_dir (Optional[str]): Base directory for outputs. Defaults to current directory.
            log_level (int): Logging level. Defaults to logging.INFO.
        """
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path.cwd()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _setup_logging(log_level: int) -> None:
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # CSV Operations
    def save_to_csv(self, df: pd.DataFrame, filepath: str, 
                    use_timestamp: bool = False, **kwargs) -> Path:
        """
        Save a DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save.
            filepath (str): Destination file path.
            use_timestamp (bool): Whether to add timestamp to filename.
            **kwargs: Additional arguments passed to pd.DataFrame.to_csv()
        
        Returns:
            Path: The actual path where the file was saved.
        """
        path = Path(filepath)
        
        if use_timestamp:
            base_name = path.stem
            extension = path.suffix
            timestamped_name = self.generate_timestamped_filename(base_name, extension)
            path = path.parent / timestamped_name
        
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
        self.logger.info(f"CSV saved to: {path}")
        return path
    
    def load_from_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load a DataFrame from a CSV file.
        
        Args:
            filepath (str): Source file path.
            **kwargs: Additional arguments passed to pd.read_csv()
        
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        df = pd.read_csv(path, **kwargs)
        self.logger.info(f"CSV loaded from: {path} (shape: {df.shape})")
        return df
    
    # Model Serialization
    def serialize_model(self, model: Any, path: str, metadata: Optional[Dict] = None) -> Path:
        """
        Serialize (save) a model object to disk using pickle.
        
        Args:
            model (Any): Trained model.
            path (str): File path to save the model.
            metadata (Optional[Dict]): Optional metadata to save alongside model.
        
        Returns:
            Path: The path where the model was saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as file:
            pickle.dump(model, file)
        
        if metadata:
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"Model metadata saved to: {metadata_path}")
        
        self.logger.info(f"Model serialized to: {path}")
        return path
    
    def deserialize_model(self, path: str) -> Any:
        """
        Deserialize (load) a model object from disk.
        
        Args:
            path (str): File path to load the model from.
        
        Returns:
            Any: Loaded model object.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as file:
            model = pickle.load(file)
        
        self.logger.info(f"Model deserialized from: {path}")
        return model
    
    # Logging Functions
    def log_pipeline_step(self, step_name: str, status: str, 
                         extra_info: Optional[Dict] = None) -> None:
        """
        Log a pipeline step for monitoring purposes.
        
        Args:
            step_name (str): Name of the step.
            status (str): Status message (e.g., 'started', 'completed', 'failed').
            extra_info (Optional[Dict]): Additional information to log.
        """
        message = f"Step '{step_name}' - Status: {status}"
        
        if extra_info:
            info_str = ", ".join(f"{k}={v}" for k, v in extra_info.items())
            message += f" | {info_str}"
        
        if status.lower() in ['failed', 'error']:
            self.logger.error(message)
        elif status.lower() == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    # File Naming Utilities
    @staticmethod
    def generate_timestamped_filename(base_name: str, extension: str = ".csv") -> str:
        """
        Generate a timestamped filename with a given base name and extension.
        
        Args:
            base_name (str): The base name of the file (without extension).
            extension (str, optional): The file extension to append. Defaults to '.csv'.
        
        Returns:
            str: The generated filename including the timestamp and extension.
        
        Raises:
            TypeError: If 'base_name' or 'extension' is not a string.
        """
        if not isinstance(base_name, str) or not isinstance(extension, str):
            raise TypeError("Base name and extension must be strings")
        
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{base_name}_{timestamp}{extension}"
    
    # Additional Utility Functions
    def save_to_json(self, data: Union[Dict, list], filepath: str, 
                     use_timestamp: bool = False, **kwargs) -> Path:
        """
        Save data to a JSON file.
        
        Args:
            data (Union[Dict, list]): Data to save.
            filepath (str): Destination file path.
            use_timestamp (bool): Whether to add timestamp to filename.
            **kwargs: Additional arguments passed to json.dump()
        
        Returns:
            Path: The actual path where the file was saved.
        """
        path = Path(filepath)
        
        if use_timestamp:
            base_name = path.stem
            extension = path.suffix
            timestamped_name = self.generate_timestamped_filename(base_name, extension)
            path = path.parent / timestamped_name
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str, **kwargs)
        
        self.logger.info(f"JSON saved to: {path}")
        return path
    
    def load_from_json(self, filepath: str) -> Union[Dict, list]:
        """
        Load data from a JSON file.
        
        Args:
            filepath (str): Source file path.
        
        Returns:
            Union[Dict, list]: Loaded data.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"JSON loaded from: {path}")
        return data
    
    def compute_file_hash(self, filepath: str, algorithm: str = 'sha256') -> str:
        """
        Compute hash of a file for integrity checking.
        
        Args:
            filepath (str): Path to the file.
            algorithm (str): Hash algorithm to use. Defaults to 'sha256'.
        
        Returns:
            str: Hexadecimal hash string.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        hash_obj = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        hash_value = hash_obj.hexdigest()
        self.logger.info(f"Computed {algorithm} hash for {path}: {hash_value}")
        return hash_value
    
    def create_pipeline_manifest(self, manifest_data: Dict, 
                                 filepath: Optional[str] = None) -> Path:
        """
        Create a manifest file documenting pipeline execution.
        
        Args:
            manifest_data (Dict): Manifest information (e.g., steps, timestamps, file paths).
            filepath (Optional[str]): Custom path for manifest. Defaults to timestamped file.
        
        Returns:
            Path: Path where manifest was saved.
        """
        if filepath is None:
            filepath = self.base_output_dir / self.generate_timestamped_filename(
                "pipeline_manifest", ".json"
            )
        
        manifest_data['created_at'] = datetime.now().isoformat()
        return self.save_to_json(manifest_data, str(filepath))
    
    def get_directory_size(self, dirpath: str) -> int:
        """
        Calculate total size of a directory in bytes.
        
        Args:
            dirpath (str): Directory path.
        
        Returns:
            int: Total size in bytes.
        """
        path = Path(dirpath)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        self.logger.info(f"Directory {path} total size: {total_size / (1024**2):.2f} MB")
        return total_size
