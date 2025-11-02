import logging
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool_decider import get_tool_decider, create_data_summary, ToolDecider

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class DynamicAnalysisAgent:
    """
    Rule-based SLM for dynamic analysis tool selection.
    Chooses between multiple classifiers and anomaly detection based on data/task.
    """
    def __init__(self, data: pd.DataFrame, target_column: Optional[str] = None, task: str = "classification", 
                 params: Dict[str, Any] = None, tool_decider: Optional[ToolDecider] = None):
        self.data = data
        self.target_column = target_column
        self.task = task
        self.params = params or {}
        self.tool_decider = tool_decider or get_tool_decider("rule_based")
        self.model = None
        self.model_name = None
        self.results = {}
        self.tried_models = []  # Track models already tried
        self.best_performance = -float('inf')  # Track best performance
        self.best_model = None
        self.best_results = None
        logging.info(f"DynamicAnalysisAgent initialized for task: {task} with params: {self.params}")

    def choose_tool(self) -> str:
        """
        LLM-based selection of analysis tool using ToolDecider.
        Returns the name of the chosen tool.
        """
        if self.task == "anomaly_detection":
            self.model_name = "IsolationForest"
            logging.info(f"Anomaly detection task, selected tool: {self.model_name}")
        else:
            # Use ToolDecider for model selection
            data_summary = create_data_summary(self.data)
            available_models = ["RandomForestClassifier", "LogisticRegression", "SVC", 
                              "RandomForestRegressor", "LinearRegression", "Ridge", "Lasso", "SVR"]
            
            decision = self.tool_decider.decide_model_family(self.task, data_summary, available_models)
            self.model_name = decision.get("model", "RandomForestClassifier")
            logging.info(f"ToolDecider selected tool: {self.model_name}, reason: {decision.get('reason', 'N/A')}")
            
        return self.model_name

    def run(self, force_retry: bool = False) -> Dict[str, Any]:
        """
        Executes the chosen analysis tool and returns results.
        If performance is poor and force_retry=True, try multiple models.
        """
        if self.task == "anomaly_detection":
            return self._run_anomaly_detection()
            
        if self.target_column is None:
            logging.error("Target column required for supervised learning tasks.")
            return None
        
        # If force_retry is True, try multiple models to find the best one
        if force_retry:
            return self._try_multiple_models()
            
        tool = self.choose_tool()
        if tool == "LogisticRegression":
            return self._run_logistic_regression()
        elif tool == "SVC":
            return self._run_svc()
        elif tool == "RandomForestRegressor":
            return self._run_random_forest_regressor()
        elif tool == "LinearRegression":
            return self._run_linear_regression()
        elif tool == "Ridge":
            return self._run_ridge()
        elif tool == "Lasso":
            return self._run_lasso()
        elif tool == "SVR":
            return self._run_svr()
        else:
            return self._run_random_forest()

    def _try_multiple_models(self) -> Optional[Dict[str, Any]]:
        """
        Try multiple models and return the best performing one.
        """
        logging.info("🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...")
        
        # Define model candidates based on task type
        if self.task == "classification":
            model_candidates = [
                ("RandomForestClassifier", self._run_random_forest),
                ("LogisticRegression", self._run_logistic_regression),
                ("SVC", self._run_svc)
            ]
        elif self.task == "regression":
            model_candidates = [
                ("RandomForestRegressor", self._run_random_forest_regressor),
                ("LinearRegression", self._run_linear_regression),
                ("Ridge", self._run_ridge),
                ("Lasso", self._run_lasso),
                ("SVR", self._run_svr)
            ]
        else:
            return None
        
        best_performance = -float('inf')
        best_model_name = None
        best_results = None
        
        for model_name, model_func in model_candidates:
            if model_name in self.tried_models:
                logging.info(f"⏭️ Skipping {model_name} (already tried)")
                continue
                
            try:
                logging.info(f"🔄 Trying {model_name}...")
                results = model_func()
                
                if results:
                    # Calculate performance metric
                    if self.task == "classification":
                        performance = results.get("accuracy", 0)
                    else:  # regression
                        performance = results.get("r2", -float('inf'))
                    
                    logging.info(f"   {model_name} performance: {performance:.4f}")
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_model_name = model_name
                        best_results = results
                        self.best_model = model_name
                        self.best_results = results
                        self.best_performance = performance
                
                self.tried_models.append(model_name)
                
            except Exception as e:
                logging.warning(f"   {model_name} failed: {str(e)}")
                self.tried_models.append(model_name)
                continue
        
        if best_results and best_performance > -float('inf'):
            logging.info(f"🏆 Best model: {best_model_name} (performance: {best_performance:.4f})")
            return best_results
        else:
            logging.error("❌ All models failed or produced invalid results")
            return None

    def _run_random_forest(self) -> Dict[str, Any]:
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        logging.info(f"Random Forest accuracy: {acc:.4f}")
        return {
            "model": "RandomForestClassifier",
            "accuracy": acc,
            "classification_report": report,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_importances": self.model.feature_importances_,
            "feature_names": X.columns.tolist()
        }

    def _run_logistic_regression(self) -> Dict[str, Any]:
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        logging.info(f"Logistic Regression accuracy: {acc:.4f}")
        return {
            "model": "LogisticRegression",
            "accuracy": acc,
            "classification_report": report,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test
        }

    def _run_svc(self) -> Dict[str, Any]:
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = SVC()
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        logging.info(f"SVM accuracy: {acc:.4f}")
        return {
            "model": "SVC",
            "accuracy": acc,
            "classification_report": report,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test
        }

    def _run_anomaly_detection(self) -> Dict[str, Any]:
        # For anomaly detection, use all features (no target column to drop)
        # But drop ID columns as they shouldn't be used for modeling
        X = self.data.copy()
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from anomaly detection: {id_columns}")
            X = X.drop(columns=id_columns)
        # Configure IsolationForest using provided params or defaults
        cont = self.params.get('contamination') if getattr(self, 'params', None) else None
        n_est = self.params.get('n_estimators') if getattr(self, 'params', None) else None

        # Defaults
        if cont is None:
            cont = 0.1
        if n_est is None:
            n_est = 200

        # If user/LLM suggested 'auto' for contamination, estimate it from data
        if isinstance(cont, str) and cont == 'auto':
            # Quick estimation: fit a temporary IsolationForest to get scores and pick a percentile
            temp_if = IsolationForest(random_state=42, n_estimators=64, max_samples='auto')
            temp_if.fit(X)
            scores = temp_if.score_samples(X)
            # lower scores => more anomalous; estimate contamination as fraction below 5th percentile
            thresh = np.percentile(scores, 5)
            cont_est = float((scores < thresh).mean())
            # cap the estimated contamination to a reasonable maximum (20%) to avoid over-flagging
            cont = min(max(cont_est, 0.001), 0.2)
            logging.info(f"Estimated contamination (auto) = {cont:.4f}")

        # Higher n_estimators for stability if provided
        self.model = IsolationForest(
            random_state=42,
            contamination=cont,
            n_estimators=int(n_est),
            max_samples='auto'
        )
        
        # Fit and predict anomalies
        preds = self.model.fit_predict(X)
        anomaly_scores = self.model.score_samples(X)
        n_anomalies = sum(preds == -1)
        
        # Create anomaly detection results DataFrame
        # Use self.data for ID columns since X has ID columns dropped
        # Handle both original column names and renamed pass-through columns (remainder__Machine_ID)
        machine_id_col = None
        for col in self.data.columns:
            if 'MACHINE_ID' in col.upper():
                machine_id_col = col
                break
        
        if machine_id_col is None:
            logging.warning(f"Machine_ID column not found in self.data. Columns available: {self.data.columns.tolist()}")
        
        results_df = pd.DataFrame({
            'Timestamp': self.data['Timestamp'] if 'Timestamp' in self.data else pd.Series(range(len(X))),
            'Machine_ID': self.data[machine_id_col] if machine_id_col else pd.Series(['Unknown'] * len(X)),
            'Anomaly_Score': anomaly_scores,
            'Is_Anomaly': preds == -1
        })
        
        # Add key metrics for anomalous points
        for col in X.columns:
            if col not in ['Timestamp', 'Machine_ID'] and X[col].dtype in ['int64', 'float64']:
                results_df[f'{col}_Value'] = X[col]
                results_df[f'{col}_zscore'] = (X[col] - X[col].mean()) / X[col].std()
        
        # Sort by anomaly score (most anomalous first)
        results_df = results_df.sort_values('Anomaly_Score')
        
        logging.info(f"Isolation Forest detected {n_anomalies} anomalies out of {len(preds)} samples ({(n_anomalies/len(preds)*100):.1f}%).")
        
        return {
            "model": "IsolationForest",
            "anomaly_labels": preds,
            "feature_names": X.columns.tolist(),
            "n_anomalies": n_anomalies,
            "anomaly_scores": anomaly_scores,
            "results_df": results_df,  # Detailed results for optimization
            "X": X  # Original data for context
        }

    def _run_random_forest_regressor(self) -> Dict[str, Any]:
        """Run Random Forest Regressor."""
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use ToolDecider for hyperparameters if not provided
        if not self.params:
            data_summary = create_data_summary(self.data)
            hyperparams = self.tool_decider.decide_hyperparameters("RandomForestRegressor", "regression", data_summary)
            n_estimators = hyperparams.get('n_estimators', 100)
            max_depth = hyperparams.get('max_depth', None)
            random_state = hyperparams.get('random_state', 42)
            logging.info(f"Using ToolDecider hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")
        else:
            n_estimators = self.params.get('n_estimators', 100)
            max_depth = self.params.get('max_depth', None)
            random_state = self.params.get('random_state', 42)
        
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logging.info(f"Random Forest Regressor - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            "model": "RandomForestRegressor",
            "mse": mse,
            "r2": r2,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_importances": self.model.feature_importances_,
            "feature_names": X.columns.tolist()
        }

    def _run_linear_regression(self) -> Dict[str, Any]:
        """Run Linear Regression."""
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logging.info(f"Linear Regression - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            "model": "LinearRegression",
            "mse": mse,
            "r2": r2,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X.columns.tolist()
        }

    def _run_ridge(self) -> Dict[str, Any]:
        """Run Ridge Regression."""
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        alpha = self.params.get('alpha', 1.0)
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logging.info(f"Ridge Regression - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            "model": "Ridge",
            "mse": mse,
            "r2": r2,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X.columns.tolist()
        }

    def _run_lasso(self) -> Dict[str, Any]:
        """Run Lasso Regression."""
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        alpha = self.params.get('alpha', 1.0)
        self.model = Lasso(alpha=alpha)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logging.info(f"Lasso Regression - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            "model": "Lasso",
            "mse": mse,
            "r2": r2,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X.columns.tolist()
        }

    def _run_svr(self) -> Dict[str, Any]:
        """Run Support Vector Regression."""
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        kernel = self.params.get('kernel', 'rbf')
        C = self.params.get('C', 1.0)
        self.model = SVR(kernel=kernel, C=C)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        train_preds = self.model.predict(X_train)
        
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logging.info(f"Support Vector Regression - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            "model": "SVR",
            "mse": mse,
            "r2": r2,
            "predictions": preds,
            "train_predictions": train_preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X.columns.tolist()
        }

if __name__ == "__main__":
    # Example usage
    logging.info("--- Running DynamicAnalysisAgent in Standalone Mode ---")
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1,2,3,4,5,6,7,8,9,10],
        'B': [2,3,4,5,6,7,8,9,10,11],
        'target': [0,1,0,1,0,1,0,1,0,1]
    })
    agent = DynamicAnalysisAgent(df, target_column='target')
    results = agent.run()
    logging.info(f"Results: {results}")
