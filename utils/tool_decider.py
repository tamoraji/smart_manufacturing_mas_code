"""
ToolDecider - Mini-LLM interface for agent-level tool selection decisions.
This module provides a standardized interface for small LLMs to make tactical decisions
about preprocessing tools, model selection, and hyperparameters.
"""

import logging
import json
import pandas as pd
import sys
import os
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.local_llm_agent import LocalLLMAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class ToolDecider(ABC):
    """Abstract base class for tool decision making."""
    
    @abstractmethod
    def decide_preprocessing_strategy(self, data_summary: Dict[str, Any], 
                                    available_tools: List[str]) -> Dict[str, Any]:
        """
        Decide on preprocessing strategy based on data characteristics.
        Returns:
            dict: {"strategy": "strategy_name", "tools": ["tool1", "tool2"], "reason": "..."}
        """
        pass
    
    @abstractmethod
    def decide_model_family(self, task_type: str, data_summary: Dict[str, Any],
                          available_models: List[str]) -> Dict[str, Any]:
        """
        Decide on model family based on task and data characteristics.
        Returns:
            dict: {"model": "model_name", "reason": "..."}
        """
        pass
    
    @abstractmethod
    def decide_hyperparameters(self, model: str, task_type: str, 
                             data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on hyperparameters for the chosen model.
        Returns:
            dict: {"param1": value1, "param2": value2, "reason": "..."}
        """
        pass

class RuleBasedToolDecider(ToolDecider):
    """Rule-based tool decider (fallback when no LLM available)."""
    
    def decide_preprocessing_strategy(self, data_summary: Dict[str, Any], 
                                    available_tools: List[str]) -> Dict[str, Any]:
        """Rule-based preprocessing strategy selection."""
        missing_pct = data_summary.get('missing_percentage', 0)
        numeric_count = data_summary.get('numeric_columns', 0)
        categorical_count = data_summary.get('categorical_columns', 0)
        
        strategy = "standard"
        tools = []
        
        if missing_pct > 10:
            tools.append("imputation")
        if numeric_count > 0:
            tools.append("scaling")
        if categorical_count > 0:
            tools.append("encoding")
        
        if not tools:
            tools = ["passthrough"]
            
        return {
            "strategy": strategy,
            "tools": tools,
            "reason": f"Rule-based selection: {missing_pct:.1f}% missing, {numeric_count} numeric, {categorical_count} categorical columns"
        }
    
    def decide_model_family(self, task_type: str, data_summary: Dict[str, Any],
                          available_models: List[str]) -> Dict[str, Any]:
        """Rule-based model family selection."""
        n_samples = data_summary.get('n_samples', 0)
        n_features = data_summary.get('n_features', 0)
        
        if task_type == "classification":
            if n_samples < 1000:
                model = "LogisticRegression"
            elif n_features > 50:
                model = "RandomForestClassifier"
            else:
                model = "RandomForestClassifier"
        elif task_type == "regression":
            if n_samples < 1000:
                model = "LinearRegression"
            else:
                model = "RandomForestRegressor"
        elif task_type == "anomaly_detection":
            model = "IsolationForest"
        else:
            model = "RandomForestClassifier"
            
        return {
            "model": model,
            "reason": f"Rule-based selection for {task_type}: {n_samples} samples, {n_features} features"
        }
    
    def decide_hyperparameters(self, model: str, task_type: str, 
                             data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based hyperparameter selection."""
        n_samples = data_summary.get('n_samples', 1000)
        
        if model == "RandomForestClassifier":
            return {
                "n_estimators": min(100, max(10, n_samples // 10)),
                "max_depth": None,
                "random_state": 42,
                "reason": f"Adaptive n_estimators based on {n_samples} samples"
            }
        elif model == "IsolationForest":
            return {
                "contamination": "auto",
                "n_estimators": 200,
                "random_state": 42,
                "reason": "Standard settings for anomaly detection"
            }
        elif model == "LogisticRegression":
            return {
                "max_iter": 1000,
                "random_state": 42,
                "reason": "Standard settings for logistic regression"
            }
        else:
            return {
                "random_state": 42,
                "reason": "Default settings"
            }

class LLMToolDecider(ToolDecider):
    """LLM-based tool decider using small local models."""
    
    def __init__(self, llm_agent: Optional[LocalLLMAgent] = None, 
                 fallback_decider: Optional[ToolDecider] = None):
        """
        Initialize LLM tool decider.
        Args:
            llm_agent: Local LLM agent for decisions (if None, uses fallback)
            fallback_decider: Fallback decider when LLM fails
        """
        self.llm_agent = llm_agent
        self.fallback_decider = fallback_decider or RuleBasedToolDecider()
        
    def _query_llm(self, prompt: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query the LLM with structured prompt and parse response."""
        try:
            if self.llm_agent is None:
                return None
                
            full_prompt = f"{prompt}\n\nContext: {json.dumps(context, indent=2)}\n\nRespond with valid JSON only."
            response = self.llm_agent.generate(full_prompt, max_tokens=512)
            
            # Try to extract JSON from response
            parsed = response.get('parsed')
            if parsed:
                return parsed
            
            # Try to parse raw response
            raw_text = response.get('raw', '')
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end > start:
                json_str = raw_text[start:end+1]
                return json.loads(json_str)
                
        except Exception as e:
            logging.warning(f"LLM query failed: {e}")
            
        return None
    
    def decide_preprocessing_strategy(self, data_summary: Dict[str, Any], 
                                    available_tools: List[str]) -> Dict[str, Any]:
        """Use LLM to decide preprocessing strategy."""
        prompt = """You are a data preprocessing expert. Based on the dataset characteristics, decide on the best preprocessing strategy.

Available tools: imputation, scaling, encoding, normalization, feature_selection, outlier_detection

Respond with JSON: {"strategy": "strategy_name", "tools": ["tool1", "tool2"], "reason": "explanation"}"""
        
        context = {
            "data_summary": data_summary,
            "available_tools": available_tools
        }
        
        result = self._query_llm(prompt, context)
        if result:
            return result
        else:
            logging.info("LLM preprocessing decision failed, using fallback")
            return self.fallback_decider.decide_preprocessing_strategy(data_summary, available_tools)
    
    def decide_model_family(self, task_type: str, data_summary: Dict[str, Any],
                          available_models: List[str]) -> Dict[str, Any]:
        """Use LLM to decide model family."""
        prompt = f"""You are a machine learning expert. For a {task_type} task, choose the best model family based on the dataset characteristics.

Available models: {', '.join(available_models)}

Consider: sample size, feature count, data quality, computational efficiency.

Respond with JSON: {{"model": "model_name", "reason": "explanation"}}"""
        
        context = {
            "task_type": task_type,
            "data_summary": data_summary,
            "available_models": available_models
        }
        
        result = self._query_llm(prompt, context)
        if result:
            return result
        else:
            logging.info("LLM model decision failed, using fallback")
            return self.fallback_decider.decide_model_family(task_type, data_summary, available_models)
    
    def decide_hyperparameters(self, model: str, task_type: str, 
                             data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to decide hyperparameters."""
        prompt = f"""You are a hyperparameter tuning expert. For {model} on a {task_type} task, suggest appropriate hyperparameters.

Consider: sample size, feature count, model complexity, computational constraints.

Respond with JSON: {{"param1": value1, "param2": value2, "reason": "explanation"}}

Common parameters:
- RandomForest: n_estimators, max_depth, min_samples_split
- IsolationForest: contamination, n_estimators
- LogisticRegression: max_iter, C, penalty"""
        
        context = {
            "model": model,
            "task_type": task_type,
            "data_summary": data_summary
        }
        
        result = self._query_llm(prompt, context)
        if result:
            return result
        else:
            logging.info("LLM hyperparameter decision failed, using fallback")
            return self.fallback_decider.decide_hyperparameters(model, task_type, data_summary)

def get_tool_decider(decider_type: str = "rule_based", 
                    llm_agent: Optional[LocalLLMAgent] = None,
                    **kwargs) -> ToolDecider:
    """
    Factory function to get tool decider instance.
    Args:
        decider_type: Type of decider ("rule_based", "llm", "hybrid")
        llm_agent: LLM agent for LLM-based decider
        **kwargs: Additional arguments
    Returns:
        ToolDecider instance
    """
    if decider_type.lower() == "rule_based":
        return RuleBasedToolDecider()
    elif decider_type.lower() == "llm":
        return LLMToolDecider(llm_agent=llm_agent)
    elif decider_type.lower() == "hybrid":
        # Try LLM first, fallback to rules
        return LLMToolDecider(llm_agent=llm_agent, fallback_decider=RuleBasedToolDecider())
    else:
        raise ValueError(f"Unknown tool decider type: {decider_type}")

def create_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary of dataset characteristics for tool decision making.
    Args:
        df: Input DataFrame
    Returns:
        dict: Data summary
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        "n_samples": int(len(df)),
        "n_features": int(len(df.columns)),
        "numeric_columns": int(len(numeric_cols)),
        "categorical_columns": int(len(categorical_cols)),
        "missing_percentage": float(df.isnull().mean().mean() * 100),
        "has_missing_values": bool(df.isnull().any().any()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "dtypes": df.dtypes.apply(lambda x: str(x.name)).to_dict(),
        "numeric_cols": [str(col) for col in numeric_cols],
        "categorical_cols": [str(col) for col in categorical_cols]
    }

if __name__ == "__main__":
    # Test the tool decider
    import pandas as pd
    
    # Create test data
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Create data summary
    summary = create_data_summary(test_data)
    print("Data Summary:")
    print(json.dumps(summary, indent=2))
    
    # Test rule-based decider
    print("\n=== Testing Rule-based Decider ===")
    rule_decider = RuleBasedToolDecider()
    
    preprocessing = rule_decider.decide_preprocessing_strategy(
        summary, ["imputation", "scaling", "encoding"]
    )
    print("Preprocessing decision:", json.dumps(preprocessing, indent=2))
    
    model = rule_decider.decide_model_family(
        "classification", summary, ["RandomForestClassifier", "LogisticRegression"]
    )
    print("Model decision:", json.dumps(model, indent=2))
    
    hyperparams = rule_decider.decide_hyperparameters(
        "RandomForestClassifier", "classification", summary
    )
    print("Hyperparameters:", json.dumps(hyperparams, indent=2))
    
    # Test LLM decider (with mock LLM)
    print("\n=== Testing LLM Decider (with mock) ===")
    mock_llm = LocalLLMAgent(backend='mock')
    llm_decider = LLMToolDecider(llm_agent=mock_llm)
    
    preprocessing = llm_decider.decide_preprocessing_strategy(
        summary, ["imputation", "scaling", "encoding"]
    )
    print("LLM Preprocessing decision:", json.dumps(preprocessing, indent=2))
