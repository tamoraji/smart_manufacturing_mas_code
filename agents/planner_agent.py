import logging
import pandas as pd
from agents.data_loader_agent import DataLoaderAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.analysis_agent import AnalysisAgent
from agents.optimization_agent import OptimizationAgent

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class PlannerAgent:
    """
    The PlannerAgent orchestrates the entire MAS workflow, from data loading to analysis.
    It acts as the central "brain" of the system, managing the sequence of operations
    and the flow of data between other agents.
    """
    def __init__(self, dataset_path: str, target_column: str):
        """
        Initialize the PlannerAgent.
        Args:
            dataset_path (str): The path to the dataset to be processed.
            target_column (str): The name of the target variable for analysis.
        """
        logging.info("Initializing Planner Agent...")
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.raw_data = None
        self.preprocessed_data = None
        self.analysis_results = None
        self.recommendations = None

    def run_workflow(self):
        """
        Executes the full MAS workflow in a predefined sequence.
        This method simulates the decision-making process of a high-level planner.
        """
        logging.info("--- Starting MAS Workflow ---")

        # Step 1: Perception Layer - Load and inspect data
        if not self._execute_perception_step():
            logging.error("Workflow halted due to failure in Perception Layer.")
            return

        # Step 2: Preprocessing Layer - Clean and prepare data
        if not self._execute_preprocessing_step():
            logging.error("Workflow halted due to failure in Preprocessing Layer.")
            return

        # Step 3: Analysis Layer - Train model and generate insights
        if not self._execute_analysis_step():
            logging.error("Workflow halted due to failure in Analysis Layer.")
            return

        # Step 4: Optimization Layer - Generate prescriptive actions
        if not self._execute_optimization_step():
            logging.warning("Optimization step failed or produced no recommendations.")

        logging.info("--- MAS Workflow Finished Successfully ---")
        logging.info("Final insights and predictions are now available for the Optimization Layer.")

    def _execute_perception_step(self) -> bool:
        """Runs the data loading and inspection agent."""
        logging.info("[Planner] Executing Perception Layer...")
        data_loader = DataLoaderAgent(self.dataset_path)
        self.raw_data = data_loader.load_data()
        if self.raw_data is None:
            return False
        data_loader.inspect_data()
        logging.info("[Planner] Perception Layer completed.")
        return True

    def _execute_preprocessing_step(self) -> bool:
        """Runs the data preprocessing agent."""
        logging.info("[Planner] Executing Preprocessing Layer...")
        if self.target_column not in self.raw_data.columns:
            logging.error(f"Target column '{self.target_column}' not in raw data.")
            return False
        
        target_data = self.raw_data[[self.target_column]]
        feature_data = self.raw_data.drop(columns=[self.target_column])
        
        preprocessing_agent = PreprocessingAgent(feature_data)
        preprocessed_features = preprocessing_agent.preprocess()
        
        if preprocessed_features is None:
            return False
            
        self.preprocessed_data = pd.concat([preprocessed_features, target_data], axis=1)
        logging.info("[Planner] Preprocessing Layer completed.")
        return True

    def _execute_analysis_step(self) -> bool:
        """Runs the data analysis agent."""
        logging.info("[Planner] Executing Analysis Layer...")
        analysis_agent = AnalysisAgent(self.preprocessed_data, target_column=self.target_column)
        analysis_agent.train_model()
        
        eval_results = analysis_agent.evaluate_model()
        feature_importances = analysis_agent.get_feature_importance()
        
        if eval_results is None or feature_importances is None:
            return False
            
        # Store all results from the analysis for the optimization agent
        self.analysis_results = {
            'evaluation': eval_results,
            'feature_importances': feature_importances,
            'test_data_features': analysis_agent.X_test, # Preprocessed features
            'test_predictions': eval_results['predictions']
        }
            
        logging.info("[Planner] Analysis Layer completed.")
        return True

    def _execute_optimization_step(self) -> bool:
        """Runs the optimization agent to generate a prescriptive action plan."""
        logging.info("[Planner] Executing Optimization Layer...")
        if self.analysis_results is None:
            logging.error("Cannot run optimization without analysis results.")
            return False
        
        # Use the index from the test set features to get the original, unprocessed data for context
        original_test_context = self.raw_data.loc[self.analysis_results['test_data_features'].index]
        
        optimization_payload = {
            'test_data': original_test_context, # Original data for context
            'test_predictions': self.analysis_results['test_predictions'],
            'feature_importances': self.analysis_results['feature_importances']
        }
        
        optimization_agent = OptimizationAgent(optimization_payload)
        self.recommendations = optimization_agent.generate_recommendations()
        
        if self.recommendations is None:
            return False # Agent failed internally
            
        logging.info("[Planner] Optimization Layer completed.")
        return True

if __name__ == '__main__':
    # Example Usage for demonstration
    logging.info("--- Running Planner Agent in Standalone Mode ---")
    
    # Define the path to the dataset for the standalone run
    DATASET_PATH = "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
    TARGET_COLUMN = "Maintenance_Priority"
    
    # Initialize and run the planner
    planner = PlannerAgent(dataset_path=DATASET_PATH, target_column=TARGET_COLUMN)
    planner.run_workflow()
    
    logging.info("--- End of Planner Agent Standalone Run ---")
