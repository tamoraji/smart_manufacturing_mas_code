"""
Data Loader Agent for Smart Manufacturing Maintenance Dataset
-----------------------------------------------------------
This agent is responsible for loading and inspecting the dataset, providing detailed reporting for educational and scientific purposes.
"""


import pandas as pd
import os
from typing import Optional, Dict, Any
import logging

# Configure logging for the entire module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class DataLoaderAgent:
    """
    DataLoaderAgent loads and inspects a CSV dataset for smart manufacturing applications.
    It uses the logging module for all output, suitable for scientific and educational use.
    """
    def __init__(self, dataset_path: str):
        """
        Initialize the DataLoaderAgent.
        Args:
            dataset_path (str): Path to the dataset CSV file.
        """
        logging.info("Initializing Data Loader Agent...")
        self.dataset_path = dataset_path
        self.data: Optional[pd.DataFrame] = None
        logging.info(f"Dataset path set to: {self.dataset_path}")

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the dataset from the specified path.
        Returns:
            pd.DataFrame or None: Loaded data or None if loading fails.
        """
        logging.info("Attempting to load the dataset...")
        if not os.path.exists(self.dataset_path):
            logging.error(f"Dataset file not found at: {self.dataset_path}")
            return None
        try:
            self.data = pd.read_csv(self.dataset_path)
            logging.info(f"Dataset loaded successfully! Shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}", exc_info=True)
            self.data = None
        return self.data

    def inspect_data(self, verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        Inspects the loaded dataset and logs a detailed report.
        Args:
            verbose (bool): If True, logs the report. If False, only returns the results.
        Returns:
            dict or None: Inspection results (rows, columns, dtypes, etc.) or None if no data.
        """
        if self.data is None:
            logging.warning("No data loaded. Cannot inspect. Please run load_data() first.")
            return None
            
        report = {
            "num_rows": self.data.shape[0],
            "num_columns": self.data.shape[1],
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.apply(lambda x: x.name).to_dict(),
            "head": self.data.head().to_string(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "describe": self.data.describe().to_string(),
        }

        if verbose:
            logging.info("--- Dataset Inspection Report ---")
            logging.info(f"Number of rows: {report['num_rows']}")
            logging.info(f"Number of columns: {report['num_columns']}")
            logging.info(f"Column names: {report['columns']}")
            logging.info(f"Data types: {report['dtypes']}")
            logging.info(f"First 5 rows:\n{report['head']}")
            logging.info(f"Missing values per column: {report['missing_values']}")
            logging.info(f"Basic statistics (numerical columns):\n{report['describe']}")
            logging.info("--- End of Inspection Report ---")
            
        return report


# For direct script usage, allow dataset path to be provided as a command-line argument for flexibility.
if __name__ == "__main__":
    import sys
    logging.info("This script can be run directly to load and inspect a dataset.")
    logging.info("Usage: python data_loader_agent.py <path_to_dataset.csv>")
    
    if len(sys.argv) < 2:
        logging.error("No dataset path provided. Please provide the path as a command-line argument.")
        logging.info("Example: python data_loader_agent.py '../data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv'")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    logging.info(f"Running in standalone mode with dataset path: {dataset_path}")
    
    agent = DataLoaderAgent(dataset_path)
    if agent.load_data() is not None:
        agent.inspect_data()
    else:
        logging.error("Standalone script execution failed because data could not be loaded.")
