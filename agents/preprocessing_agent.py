import pandas as pd
from typing import Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool_decider import get_tool_decider, create_data_summary, ToolDecider
from utils.intelligent_feature_analysis import IntelligentFeatureAnalyzer

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class PreprocessingAgent:
    """
    PreprocessingAgent is responsible for cleaning and preparing the dataset.
    It provides tools for handling missing values, scaling numerical features,
    and encoding categorical features. This agent is designed to be called
    by a higher-level planner agent (LLM/SLM) in the future.
    """
    def __init__(self, data: pd.DataFrame, tool_decider: Optional[ToolDecider] = None, 
                 target_column: Optional[str] = None, problem_type: Optional[str] = None,
                 protected_columns: Optional[list] = None):
        """
        Initialize the PreprocessingAgent.
        Args:
            data (pd.DataFrame): The raw data to be preprocessed.
            tool_decider (ToolDecider): Tool decider for preprocessing strategy selection.
            target_column (str): Name of the target column for intelligent analysis.
            problem_type (str): Type of problem for intelligent analysis.
            protected_columns (list): List of column names that should not be automatically dropped.
        """
        logging.info("Initializing Preprocessing Agent...")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        self.data = data.copy() # Work on a copy to avoid side effects
        self.tool_decider = tool_decider or get_tool_decider("rule_based")
        self.target_column = target_column
        self.problem_type = problem_type
        self.protected_columns = protected_columns or []
        self.feature_analyzer = None
        self.last_feature_insights = None  # Store last feature analysis results
        self.feature_insights = None
        logging.info(f"Preprocessing Agent initialized with data of shape: {self.data.shape}")

    def get_feature_types(self):
        """Identifies numerical and categorical features in the dataset."""
        numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.problem_type == 'anomaly_detection':
            kept_identifiers = []
            protected_set = set(self.protected_columns)

            for feature in list(numerical_features):
                upper_name = feature.upper()
                if ('ID' in upper_name or upper_name.endswith('_ID') or upper_name == 'ID') and feature in protected_set:
                    kept_identifiers.append(feature)
                    numerical_features.remove(feature)

            if kept_identifiers:
                logging.info(f"Protected identifier columns for anomaly detection: {kept_identifiers}")
                passthrough_df = self.data[kept_identifiers].copy()
                passthrough_df.columns = [f"identifier__{col}" for col in kept_identifiers]
                self.data = pd.concat([self.data, passthrough_df], axis=1)
                self.identifier_columns = [f"identifier__{col}" for col in kept_identifiers]
            else:
                self.identifier_columns = []
        else:
            self.identifier_columns = []
        logging.info(f"Identified Numerical Features: {numerical_features}")
        logging.info(f"Identified Categorical Features: {categorical_features}")
        return numerical_features, categorical_features

    def perform_intelligent_feature_analysis(self) -> Dict[str, Any]:
        """
        Perform intelligent feature analysis if target column and problem type are available.
        Returns:
            Dict[str, Any]: Feature analysis insights and recommendations
        """
        if not self.target_column or not self.problem_type or self.target_column not in self.data.columns:
            logging.info("Skipping intelligent feature analysis - target column or problem type not available")
            return {}
        
        logging.info("Performing intelligent feature analysis...")
        
        # Prepare data for analysis
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Initialize feature analyzer
        self.feature_analyzer = IntelligentFeatureAnalyzer(self.target_column, self.problem_type)
        
        # Perform analysis
        self.feature_insights = self.feature_analyzer.analyze_features(X, y)
        
        # Log summary
        logging.info("Intelligent feature analysis completed")
        logging.info(f"Analysis summary:\n{self.feature_insights['summary']}")
        
        return self.feature_insights

    def create_preprocessing_pipeline(self, numerical_features, categorical_features):
        """
        Creates a scikit-learn pipeline to handle preprocessing tasks using ToolDecider.
        This makes the preprocessing steps modular and reproducible.
        """
        logging.info("Creating preprocessing pipeline...")

        # Get data summary for tool decision
        data_summary = create_data_summary(self.data)
        available_tools = ["imputation", "scaling", "encoding", "normalization"]
        
        # Use ToolDecider to decide preprocessing strategy
        decision = self.tool_decider.decide_preprocessing_strategy(data_summary, available_tools)
        logging.info(f"ToolDecider chose preprocessing strategy: {decision}")

        # Build numerical transformer based on decision
        numerical_steps = []
        if "imputation" in decision.get("tools", []):
            # Choose imputation strategy based on data characteristics
            if data_summary["missing_percentage"] > 20:
                numerical_steps.append(('imputer', KNNImputer(n_neighbors=3)))
                logging.info("Using KNN imputation for high missing percentage")
            else:
                numerical_steps.append(('imputer', SimpleImputer(strategy='median')))
                logging.info("Using median imputation for low missing percentage")
        
        if "scaling" in decision.get("tools", []):
            # Choose scaling strategy based on data characteristics
            if data_summary["memory_usage_mb"] > 100:  # Large dataset
                numerical_steps.append(('scaler', RobustScaler()))  # More robust for outliers
                logging.info("Using RobustScaler for large dataset")
            else:
                numerical_steps.append(('scaler', StandardScaler()))  # Standard choice
                logging.info("Using StandardScaler for standard preprocessing")
        elif "normalization" in decision.get("tools", []):
            numerical_steps.append(('scaler', MinMaxScaler()))
            logging.info("Using MinMaxScaler for normalization")

        # Build categorical transformer
        categorical_steps = []
        if "imputation" in decision.get("tools", []):
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        if "encoding" in decision.get("tools", []):
            # Check cardinality of categorical features to prevent feature explosion
            high_cardinality_features = []
            low_cardinality_features = []
            
            for feature in categorical_features:
                unique_count = self.data[feature].nunique()
                if unique_count > 50:  # Threshold for high cardinality
                    high_cardinality_features.append(feature)
                    logging.warning(f"High cardinality feature '{feature}' ({unique_count} unique values) - will be dropped to prevent feature explosion")
                else:
                    low_cardinality_features.append(feature)
                    logging.info(f"Low cardinality feature '{feature}' ({unique_count} unique values) - safe for one-hot encoding")
            
            # Only encode low cardinality features
            if low_cardinality_features:
                categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50)))
            else:
                logging.warning("No categorical features suitable for encoding - all have high cardinality")

        # Create transformers
        numerical_transformer = Pipeline(steps=numerical_steps) if numerical_steps else None
        categorical_transformer = Pipeline(steps=categorical_steps) if categorical_steps else None

        # Combine preprocessing steps for numerical and categorical features
        transformers = []
        if numerical_transformer and numerical_features:
            transformers.append(('num', numerical_transformer, numerical_features))
        if categorical_transformer and categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough' # Keep other columns (if any)
        )
        
        logging.info(f"Preprocessing pipeline created successfully with {len(transformers)} transformers.")
        return preprocessor

    def preprocess(self) -> Optional[pd.DataFrame]:
        """
        Executes the full preprocessing pipeline on the data.
        Returns:
            pd.DataFrame or None: The preprocessed data, or None if an error occurs.
        """
        logging.info("Starting data preprocessing...")
        try:
            # Perform intelligent feature analysis if possible
            feature_insights = self.perform_intelligent_feature_analysis()
            self.last_feature_insights = feature_insights  # Store for reporting
            
            numerical_features, categorical_features = self.get_feature_types()
            
            # Drop identifiers or columns that should not be processed
            # This is a placeholder - an LLM could decide this based on column names/descriptions
            # But don't drop protected columns that are explicitly included as features
            cols_to_drop = [col for col in self.data.columns if 'ID' in col.upper() and col not in self.protected_columns]
            
            # Also drop high-cardinality categorical features to prevent feature explosion
            for feature in categorical_features:
                if feature not in self.protected_columns:  # Don't drop protected features
                    unique_count = self.data[feature].nunique()
                    if unique_count > 50 and not feature.startswith("identifier__"):  # Threshold for high cardinality
                        cols_to_drop.append(feature)
                        logging.warning(f"Dropping high cardinality feature '{feature}' ({unique_count} unique values)")
            
            # Use intelligent feature analysis recommendations
            if feature_insights and 'recommendations' in feature_insights:
                recommendations = feature_insights['recommendations']
                
                # Add recommended features to remove (but never remove the target column)
                for feature_rec in recommendations.get('features_to_remove', []):
                    if (feature_rec['feature'] in self.data.columns and 
                        feature_rec['feature'] != self.target_column):
                        cols_to_drop.append(feature_rec['feature'])
                        logging.info(f"Intelligently removing feature '{feature_rec['feature']}': {feature_rec['reason']}")
                
                # Log feature engineering suggestions
                for suggestion in recommendations.get('feature_engineering_suggestions', []):
                    logging.info(f"Feature engineering suggestion: {suggestion['suggestion']} - {suggestion['details']}")
            
            # Always remove target column from final processed data to prevent leakage
            if self.target_column and self.target_column in self.data.columns:
                cols_to_drop.append(self.target_column)
                logging.info(f"Removing target column '{self.target_column}' to prevent data leakage")
            
            if cols_to_drop:
                logging.info(f"Dropping identifier, high-cardinality, and redundant columns: {cols_to_drop}")
                self.data = self.data.drop(columns=cols_to_drop)
                # Re-identify features after dropping columns
                numerical_features = [f for f in numerical_features if f not in cols_to_drop]
                categorical_features = [f for f in categorical_features if f not in cols_to_drop]
            
            # Identify columns that should be kept as-is (pass-through identifiers)
            # These are protected columns that would normally be treated as categorical
            passthrough_identifiers = []
            if self.protected_columns:
                for col in self.protected_columns:
                    # Check if column exists in categorical_features and is an identifier
                    if col in categorical_features and 'ID' in col.upper():
                        passthrough_identifiers.append(col)
                        logging.info(f"Keeping identifier column '{col}' as pass-through (not encoding)")
            
            # Remove identifier columns from categorical_features so they pass through unchanged
            if passthrough_identifiers:
                categorical_features = [f for f in categorical_features if f not in passthrough_identifiers]

            pipeline = self.create_preprocessing_pipeline(numerical_features, categorical_features)
            
            logging.info("Fitting and transforming the data with the pipeline...")
            processed_data = pipeline.fit_transform(self.data)
            feature_names = pipeline.get_feature_names_out()

            # ColumnTransformer may return a sparse matrix when one-hot encoding is used.
            # Convert sparse outputs to a pandas-compatible representation without forcing densification.
            if sparse.issparse(processed_data):
                processed_df = pd.DataFrame.sparse.from_spmatrix(
                    processed_data,
                    index=self.data.index,
                    columns=feature_names
                )
            else:
                processed_df = pd.DataFrame(processed_data, columns=feature_names, index=self.data.index)
            
            logging.info(f"Data preprocessing complete. New data shape: {processed_df.shape}")
            return processed_df

        except Exception as e:
            logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
            return None

if __name__ == '__main__':
    # Example Usage for demonstration
    logging.info("--- Running Preprocessing Agent in Standalone Mode ---")
    
    # Create a sample DataFrame for demonstration
    sample_data = {
        'MachineID': ['M01', 'M02', 'M03', 'M04', 'M05'],
        'Temperature': [300.1, 301.5, 299.8, 302.1, 301.9],
        'Vibration': [1.5, 1.7, 1.4, 1.8, 1.9],
        'Failure_Type': ['None', 'Power', 'None', 'Overstrain', 'None'],
        'Downtime_Cost': [100, 5000, 90, 8000, 120]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.iloc[2, 1] = None # Introduce a missing value
    
    logging.info("Created sample dataset:")
    logging.info("\n" + sample_df.to_string())
    
    # Initialize and run the agent
    preprocessing_agent = PreprocessingAgent(sample_df)
    preprocessed_data = preprocessing_agent.preprocess()
    
    if preprocessed_data is not None:
        logging.info("--- Preprocessed Data ---")
        logging.info("\n" + preprocessed_data.to_string())
        logging.info("--- End of Standalone Run ---")

