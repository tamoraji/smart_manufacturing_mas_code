"""
CrewAI Data Intelligence Agent for Smart Manufacturing
----------------------------------------------------
This agent combines data loading, inspection, and schema discovery capabilities
for intelligent manufacturing data analysis.
"""

import pandas as pd
import os
import logging
from typing import Optional, Dict, Any, List
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.schema_discovery import discover_dataset_schema, ColumnRoleDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class DataIntelligenceAgent:
    """
    CrewAI Data Intelligence Agent for smart manufacturing data analysis.
    Combines data loading, inspection, and schema discovery capabilities.
    """
    
    def __init__(self, llm_model: str = "gemini-2.5-flash", temperature: float = 0.1):
        """
        Initialize the Data Intelligence Agent.
        
        Args:
            llm_model: LLM model to use (gemini-pro, gemini-1.5-pro, etc.)
            temperature: Temperature for LLM responses
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.agent = self._create_agent()
        
    def _initialize_llm(self):
        """Initialize the Gemini LLM for CrewAI using a different approach."""
        # Check if API key is available
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file or environment variables. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Set the API key in environment for CrewAI
        os.environ["GOOGLE_API_KEY"] = api_key
        
        try:
            # Try using CrewAI's built-in LLM initialization
            from crewai.llm import LLM
            
            # Initialize using CrewAI's LLM class
            llm = LLM(
                model="gemini-2.5-flash",
                api_key=api_key,
                temperature=self.temperature
            )
            
            logging.info("Gemini LLM initialized successfully using CrewAI LLM class")
            return llm
            
        except Exception as e:
            logging.warning(f"CrewAI LLM class failed: {e}, trying direct approach")
            
            try:
                # Fallback to direct ChatGoogleGenerativeAI
                base_llm = ChatGoogleGenerativeAI(
                    model=self.llm_model, 
                    temperature=self.temperature
                )
                
                # Test the LLM to ensure it works
                test_response = base_llm.invoke("Hello")
                logging.info("Gemini LLM initialized successfully for CrewAI")
                
                # Create a wrapper class that includes the required method
                class CrewAICompatibleLLM:
                    def __init__(self, base_llm):
                        self._base_llm = base_llm
                        # Copy all attributes from the base LLM
                        for attr in dir(base_llm):
                            if not attr.startswith('_') and not callable(getattr(base_llm, attr)):
                                setattr(self, attr, getattr(base_llm, attr))
                    
                    def __getattr__(self, name):
                        # Delegate any missing attributes to the base LLM
                        return getattr(self._base_llm, name)
                    
                    def supports_stop_words(self):
                        """CrewAI compatibility method."""
                        return False
                    
                    def invoke(self, *args, **kwargs):
                        """Delegate invoke calls to the base LLM."""
                        return self._base_llm.invoke(*args, **kwargs)
                
                return CrewAICompatibleLLM(base_llm)
                
            except Exception as e2:
                logging.error(f"Failed to initialize Gemini LLM: {e2}")
                raise ValueError(f"Could not initialize Gemini LLM: {e2}")
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent with specific role and capabilities."""
        if self.llm is None:
            raise ValueError("LLM must be initialized before creating CrewAI agent")
        
        logging.info(f"Creating CrewAI agent with LLM type: {type(self.llm).__name__}")
        logging.info(f"LLM has supports_stop_words method: {hasattr(self.llm, 'supports_stop_words')}")
        
        return Agent(
            role="Data Intelligence Specialist",
            goal="Load, inspect, and analyze manufacturing datasets to understand their structure, quality, and characteristics",
            backstory="""You are an expert data intelligence specialist with deep expertise in manufacturing data analysis. 
            You have extensive experience with sensor data, IoT metrics, and manufacturing parameters. 
            You excel at understanding data quality, detecting patterns, and providing comprehensive insights 
            about manufacturing datasets. Your analysis helps other agents make informed decisions about 
            preprocessing and modeling strategies.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],  # We'll add custom tools later
            memory=True
        )
    
    def analyze_dataset(self, dataset_path: str, problem_type: str = None, target_column: str = None) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis combining loading, inspection, and schema discovery.
        Now problem-aware for different ML tasks.
        
        Args:
            dataset_path: Path to the CSV dataset
            problem_type: Type of problem ('classification', 'regression', 'anomaly_detection')
            target_column: Specific target column if known
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logging.info(f"Starting comprehensive dataset analysis for: {dataset_path}")
        logging.info(f"Problem type: {problem_type or 'auto-detect'}")
        logging.info(f"Target column: {target_column or 'auto-detect'}")
        
        # Step 1: Load and basic inspection
        basic_analysis = self._load_and_inspect_data(dataset_path)
        if basic_analysis is None:
            return {"error": "Failed to load dataset"}
        
        # Step 2: Problem-aware schema discovery
        schema_analysis = self._discover_schema(dataset_path, problem_type, target_column)
        
        # Step 3: Data quality assessment
        quality_analysis = self._assess_data_quality(basic_analysis['data'])
        
        # Step 4: Manufacturing-specific insights
        manufacturing_insights = self._analyze_manufacturing_patterns(basic_analysis['data'])
        
        # Step 5: Problem-specific feature analysis
        feature_analysis = self._analyze_features_for_problem(
            basic_analysis['data'], 
            schema_analysis, 
            problem_type, 
            target_column
        )
        
        # Step 6: Generate intelligent recommendations (with Gemini if available)
        recommendations = self._generate_problem_aware_recommendations(
            basic_analysis, schema_analysis, quality_analysis, 
            manufacturing_insights, feature_analysis, problem_type
        )
        
        # Combine all analyses
        comprehensive_analysis = {
            "dataset_path": dataset_path,
            "problem_type": problem_type,
            "target_column": target_column,
            "basic_info": basic_analysis,
            "schema_discovery": schema_analysis,
            "data_quality": quality_analysis,
            "manufacturing_insights": manufacturing_insights,
            "feature_analysis": feature_analysis,
            "recommendations": recommendations
        }
        
        logging.info("Comprehensive dataset analysis completed successfully")
        return comprehensive_analysis
    
    def _load_and_inspect_data(self, dataset_path: str) -> Optional[Dict[str, Any]]:
        """Load dataset and perform basic inspection."""
        try:
            if not os.path.exists(dataset_path):
                logging.error(f"Dataset file not found at: {dataset_path}")
                return None
            
            # Load data
            data = pd.read_csv(dataset_path)
            logging.info(f"Dataset loaded successfully! Shape: {data.shape}")
            
            # Basic inspection
            inspection_report = {
                "data": data,
                "num_rows": data.shape[0],
                "num_columns": data.shape[1],
                "columns": data.columns.tolist(),
                "dtypes": data.dtypes.apply(lambda x: x.name).to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
                "memory_usage": data.memory_usage(deep=True).sum(),
                "sample_data": data.head(3).to_dict('records')
            }
            
            return inspection_report
            
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}", exc_info=True)
            return None
    
    def _discover_schema(self, dataset_path: str, problem_type: str = None, target_column: str = None) -> Dict[str, Any]:
        """Discover dataset schema and column roles with problem awareness."""
        try:
            data = pd.read_csv(dataset_path)
            schema = discover_dataset_schema(data, target_column)
            
            # Add additional schema insights
            schema["column_roles"] = {}
            for col in data.columns:
                schema["column_roles"][col] = ColumnRoleDetector.detect_column_role(col, data[col])
            
            # Problem-aware target suggestions
            if problem_type:
                schema["problem_aware_targets"] = self._suggest_targets_for_problem(
                    data, schema, problem_type
                )
            
            return schema
            
        except Exception as e:
            logging.error(f"Schema discovery failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        quality_metrics = {
            "completeness": {
                "total_cells": data.size,
                "missing_cells": data.isnull().sum().sum(),
                "completeness_rate": (1 - data.isnull().sum().sum() / data.size) * 100
            },
            "uniqueness": {
                "duplicate_rows": data.duplicated().sum(),
                "duplicate_percentage": (data.duplicated().sum() / len(data)) * 100
            },
            "consistency": {
                "inconsistent_types": self._detect_inconsistent_types(data),
                "outlier_columns": self._detect_outlier_columns(data)
            },
            "validity": {
                "invalid_values": self._detect_invalid_values(data)
            }
        }
        
        return quality_metrics
    
    def _detect_inconsistent_types(self, data: pd.DataFrame) -> List[str]:
        """Detect columns with inconsistent data types."""
        inconsistent = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if numeric values are stored as strings
                numeric_count = 0
                total_count = data[col].dropna().count()
                if total_count > 0:
                    for val in data[col].dropna():
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass
                    
                    if numeric_count / total_count > 0.8:  # 80% numeric values
                        inconsistent.append(col)
        
        return inconsistent
    
    def _detect_outlier_columns(self, data: pd.DataFrame) -> List[str]:
        """Detect columns with potential outliers."""
        outlier_columns = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if len(outliers) > len(data) * 0.05:  # More than 5% outliers
                outlier_columns.append(col)
        
        return outlier_columns
    
    def _detect_invalid_values(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect invalid values in the dataset."""
        invalid_values = {}
        
        for col in data.columns:
            invalid = []
            
            # Check for negative values in columns that shouldn't have them
            if any(keyword in col.lower() for keyword in ['count', 'frequency', 'duration', 'age']):
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    invalid.append(f"Negative values: {negative_count}")
            
            # Check for unrealistic values
            if data[col].dtype in ['int64', 'float64']:
                if data[col].max() > 1e6:  # Very large values
                    invalid.append(f"Unusually large values: max={data[col].max()}")
            
            if invalid:
                invalid_values[col] = invalid
        
        return invalid_values
    
    def _analyze_manufacturing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze manufacturing-specific patterns in the data."""
        manufacturing_insights = {
            "sensor_data": self._analyze_sensor_data(data),
            "maintenance_patterns": self._analyze_maintenance_patterns(data),
            "performance_metrics": self._analyze_performance_metrics(data)
        }
        
        return manufacturing_insights
    
    def _analyze_sensor_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sensor data patterns."""
        sensor_columns = [col for col in data.columns if any(
            keyword in col.lower() for keyword in 
            ['temperature', 'pressure', 'vibration', 'acoustic', 'sensor', 'signal']
        )]
        
        sensor_analysis = {}
        for col in sensor_columns:
            if data[col].dtype in ['int64', 'float64']:
                sensor_analysis[col] = {
                    "range": [data[col].min(), data[col].max()],
                    "mean": data[col].mean(),
                    "std": data[col].std(),
                    "variation_coefficient": data[col].std() / data[col].mean() if data[col].mean() != 0 else 0
                }
        
        return sensor_analysis
    
    def _analyze_maintenance_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze maintenance-related patterns."""
        maintenance_columns = [col for col in data.columns if any(
            keyword in col.lower() for keyword in 
            ['maintenance', 'repair', 'downtime', 'failure', 'priority', 'cost']
        )]
        
        maintenance_analysis = {}
        for col in maintenance_columns:
            if data[col].dtype in ['int64', 'float64']:
                maintenance_analysis[col] = {
                    "distribution": data[col].describe().to_dict(),
                    "zero_values": (data[col] == 0).sum(),
                    "high_values": (data[col] > data[col].quantile(0.95)).sum()
                }
        
        return maintenance_analysis
    
    def _analyze_performance_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance-related metrics."""
        performance_columns = [col for col in data.columns if any(
            keyword in col.lower() for keyword in 
            ['efficiency', 'performance', 'quality', 'throughput', 'yield']
        )]
        
        performance_analysis = {}
        for col in performance_columns:
            if data[col].dtype in ['int64', 'float64']:
                performance_analysis[col] = {
                    "performance_levels": {
                        "low": (data[col] < data[col].quantile(0.33)).sum(),
                        "medium": ((data[col] >= data[col].quantile(0.33)) & 
                                 (data[col] < data[col].quantile(0.67))).sum(),
                        "high": (data[col] >= data[col].quantile(0.67)).sum()
                    },
                    "trend": "increasing" if data[col].corr(pd.Series(range(len(data)))) > 0 else "decreasing"
                }
        
        return performance_analysis
    
    def _suggest_targets_for_problem(self, data: pd.DataFrame, schema: Dict, problem_type: str) -> Dict[str, Any]:
        """Suggest target columns based on specific problem type."""
        suggestions = []
        columns = schema.get('columns', {})
        
        for col_name, col_info in columns.items():
            if col_info['role'] == 'identifier' or col_info['role'] == 'timestamp':
                continue
                
            score = 0
            reasons = []
            
            if problem_type == 'classification':
                # For classification: prefer categorical with good cardinality
                if col_info['type'] == 'categorical':
                    unique_count = col_info.get('stats', {}).get('unique_count', 0)
                    if 2 <= unique_count <= 20:
                        score += 40
                        reasons.append(f"Good cardinality for classification ({unique_count} classes)")
                    elif unique_count == 2:
                        score += 50
                        reasons.append("Binary classification target")
                elif col_info['type'] == 'numeric':
                    # Check if numeric column has discrete values suitable for classification
                    unique_count = col_info.get('stats', {}).get('unique_count', 0)
                    if 2 <= unique_count <= 10:
                        score += 30
                        reasons.append("Numeric column with discrete values suitable for classification")
                
            elif problem_type == 'regression':
                # For regression: prefer continuous numeric columns
                if col_info['type'] == 'numeric':
                    unique_count = col_info.get('stats', {}).get('unique_count', 0)
                    if unique_count > 10:
                        score += 40
                        reasons.append("Continuous numeric column suitable for regression")
                    elif unique_count > 5:
                        score += 30
                        reasons.append("Numeric column with sufficient variation for regression")
                
            elif problem_type == 'anomaly_detection':
                # For anomaly detection: prefer numeric columns with good variance
                if col_info['type'] == 'numeric':
                    stats = col_info.get('stats', {})
                    if 'std' in stats and stats['std'] > 0:
                        score += 35
                        reasons.append("Numeric column with variance suitable for anomaly detection")
                    if 'range' in stats and stats['range'] > 0:
                        score += 25
                        reasons.append("Numeric column with good range for anomaly detection")
            
            # Common scoring factors
            if col_info['role'] == 'target':
                score += 50
                reasons.append("Explicitly marked as target variable")
            
            # Check for target-like names based on problem type
            col_lower = col_name.lower()
            if problem_type == 'classification':
                if any(keyword in col_lower for keyword in ['class', 'category', 'type', 'status', 'priority', 'level']):
                    score += 30
                    reasons.append("Name suggests classification target")
            elif problem_type == 'regression':
                if any(keyword in col_lower for keyword in ['value', 'amount', 'cost', 'price', 'score', 'rating', 'duration']):
                    score += 30
                    reasons.append("Name suggests regression target")
            elif problem_type == 'anomaly_detection':
                if any(keyword in col_lower for keyword in ['anomaly', 'outlier', 'fault', 'error', 'failure']):
                    score += 30
                    reasons.append("Name suggests anomaly detection target")
            
            # Penalize high missing values
            if 'stats' in col_info and col_info['stats'].get('null_percentage', 0) > 20:
                score -= 30
                reasons.append("High missing values")
            
            if score > 0:
                suggestions.append({
                    'column': col_name,
                    'score': score,
                    'reasons': reasons,
                    'problem_type': problem_type,
                    'data_type': col_info['type'],
                    'unique_count': col_info.get('stats', {}).get('unique_count', 0)
                })
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return {
            'suggestions': suggestions[:5],
            'problem_type': problem_type,
            'total_candidates': len(suggestions)
        }
    
    def _analyze_features_for_problem(self, data: pd.DataFrame, schema: Dict, 
                                    problem_type: str, target_column: str) -> Dict[str, Any]:
        """Analyze features specifically for the given problem type."""
        analysis = {
            'problem_type': problem_type,
            'target_column': target_column,
            'feature_analysis': {},
            'preprocessing_recommendations': []
        }
        
        if not problem_type:
            return analysis
        
        # Get feature columns (exclude target, identifier, timestamp)
        feature_columns = []
        for col, info in schema.get('columns', {}).items():
            if info['role'] in ['feature'] and col != target_column:
                feature_columns.append(col)
        
        analysis['feature_columns'] = feature_columns
        analysis['feature_count'] = len(feature_columns)
        
        if problem_type == 'classification':
            analysis['preprocessing_recommendations'].extend([
                "Consider label encoding or one-hot encoding for categorical features",
                "Apply feature scaling for numeric features",
                "Check for class imbalance in target variable"
            ])
            
        elif problem_type == 'regression':
            analysis['preprocessing_recommendations'].extend([
                "Apply feature scaling (StandardScaler or MinMaxScaler)",
                "Consider polynomial features for non-linear relationships",
                "Check for feature correlation and multicollinearity"
            ])
            
        elif problem_type == 'anomaly_detection':
            analysis['preprocessing_recommendations'].extend([
                "Apply robust scaling (RobustScaler) to handle outliers",
                "Consider dimensionality reduction (PCA) for high-dimensional data",
                "Ensure features are normalized for distance-based methods"
            ])
        
        # Analyze feature types
        numeric_features = [col for col in feature_columns if schema['columns'][col]['type'] == 'numeric']
        categorical_features = [col for col in feature_columns if schema['columns'][col]['type'] == 'categorical']
        
        analysis['feature_analysis'] = {
            'numeric_features': {
                'count': len(numeric_features),
                'columns': numeric_features
            },
            'categorical_features': {
                'count': len(categorical_features),
                'columns': categorical_features
            }
        }
        
        return analysis
    
    def _generate_problem_aware_recommendations(self, basic_analysis: Dict, schema_analysis: Dict,
                                              quality_analysis: Dict, manufacturing_insights: Dict,
                                              feature_analysis: Dict, problem_type: str) -> List[str]:
        """Generate problem-aware recommendations using Gemini LLM."""
        recommendations = []
        
        # Basic recommendations
        if quality_analysis["completeness"]["completeness_rate"] < 90:
            recommendations.append("Consider imputation strategies for missing values")
        
        if quality_analysis["uniqueness"]["duplicate_percentage"] > 5:
            recommendations.append("Investigate and handle duplicate records")
        
        # Problem-specific recommendations
        if problem_type == 'classification':
            recommendations.extend([
                "For classification: Ensure target variable has balanced classes",
                "Consider stratified sampling for train/test split",
                "Evaluate feature importance for feature selection"
            ])
        elif problem_type == 'regression':
            recommendations.extend([
                "For regression: Check for linear relationships between features and target",
                "Consider feature engineering for non-linear patterns",
                "Evaluate feature correlation to avoid multicollinearity"
            ])
        elif problem_type == 'anomaly_detection':
            recommendations.extend([
                "For anomaly detection: Ensure sufficient normal samples for training",
                "Consider unsupervised methods if labeled anomalies are scarce",
                "Evaluate feature distributions for outlier detection"
            ])
        
        # Use Gemini for intelligent recommendations if available
        if self.llm:
            try:
                prompt = f"""
                Based on the following manufacturing dataset analysis, provide 3-5 specific, actionable recommendations for a {problem_type} problem:
                
                Dataset Info:
                - Shape: {basic_analysis['shape']}
                - Features: {feature_analysis.get('feature_count', 'Unknown')}
                - Data Quality: {quality_analysis['completeness']['completeness_rate']:.1f}% complete
                
                Problem Type: {problem_type}
                Target Column: {schema_analysis.get('target_column', 'Auto-detect')}
                
                Current Recommendations: {recommendations[:3]}
                
                Provide specific, actionable recommendations for this {problem_type} problem in manufacturing context.
                """
                
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    llm_recommendations = response.content.split('\n')
                    recommendations.extend([rec.strip() for rec in llm_recommendations if rec.strip()])
                else:
                    recommendations.append("LLM analysis available but response format unexpected")
                    
            except Exception as e:
                logging.warning(f"LLM recommendation generation failed: {e}")
                recommendations.append("LLM analysis temporarily unavailable")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_recommendations(self, basic_analysis: Dict, schema_analysis: Dict, 
                                quality_analysis: Dict, manufacturing_insights: Dict) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        # Data quality recommendations
        if quality_analysis["completeness"]["completeness_rate"] < 90:
            recommendations.append("Consider imputation strategies for missing values")
        
        if quality_analysis["uniqueness"]["duplicate_percentage"] > 5:
            recommendations.append("Investigate and handle duplicate records")
        
        # Schema recommendations
        if schema_analysis.get("suggested_targets"):
            recommendations.append("Use suggested target columns for modeling")
        
        # Manufacturing-specific recommendations
        if manufacturing_insights["sensor_data"]:
            recommendations.append("Apply sensor data preprocessing (scaling, outlier handling)")
        
        if manufacturing_insights["maintenance_patterns"]:
            recommendations.append("Consider maintenance cost optimization strategies")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test the agent
    agent = DataIntelligenceAgent()
    
    # Test with a sample dataset
    sample_dataset = "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
    if os.path.exists(sample_dataset):
        results = agent.analyze_dataset(sample_dataset)
        print("Analysis completed successfully!")
        print(f"Dataset shape: {results['basic_info']['num_rows']} rows, {results['basic_info']['num_columns']} columns")
    else:
        print("Sample dataset not found. Please provide a valid dataset path.")
