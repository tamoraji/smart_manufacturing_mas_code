"""
CrewAI Preprocessing Intelligence Agent for Smart Manufacturing
------------------------------------------------------------
This agent combines intelligent preprocessing, feature analysis, and data transformation
capabilities for manufacturing data analysis.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool_decider import get_tool_decider, create_data_summary, ToolDecider
from utils.intelligent_feature_analysis import IntelligentFeatureAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class PreprocessingIntelligenceAgent:
    """
    CrewAI Preprocessing Intelligence Agent for smart manufacturing data preprocessing.
    Combines intelligent feature analysis, preprocessing pipeline creation, and data transformation.
    """
    
    def __init__(self, llm_model: str = "gemini-2.5-flash", temperature: float = 0.1):
        """
        Initialize the Preprocessing Intelligence Agent.
        
        Args:
            llm_model: LLM model to use (gemini-2.5-flash, etc.)
            temperature: Temperature for LLM responses
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.agent = self._create_agent()
        
    def _initialize_llm(self):
        """Initialize the Gemini LLM."""
        try:
            # Check if API key is available
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logging.warning("GOOGLE_API_KEY not found. Agent will work without LLM insights.")
                return None
            
            # Set the API key in environment
            os.environ["GOOGLE_API_KEY"] = api_key
            
            return ChatGoogleGenerativeAI(
                model=self.llm_model, 
                temperature=self.temperature
            )
        except Exception as e:
            logging.warning(f"Failed to initialize Gemini LLM: {e}")
            return None
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent with specific role and capabilities."""
        return Agent(
            role="Data Preprocessing Specialist",
            goal="Clean, transform, and engineer features for optimal model performance in manufacturing datasets",
            backstory="""You are an expert data preprocessing specialist with deep expertise in manufacturing data transformation. 
            You excel at intelligent feature analysis, handling sensor data preprocessing, and creating robust preprocessing pipelines. 
            You understand manufacturing-specific challenges like sensor noise, maintenance patterns, and equipment performance metrics. 
            Your preprocessing strategies ensure optimal model performance while preserving critical manufacturing insights.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],  # We'll add custom tools later
            memory=True
        )
    
    def preprocess_dataset(self, data: pd.DataFrame, target_column: Optional[str] = None, 
                          problem_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive dataset preprocessing with intelligent feature analysis.
        
        Args:
            data: Input DataFrame to preprocess
            target_column: Name of target column for intelligent analysis
            problem_type: Type of problem (classification, regression, anomaly_detection)
            
        Returns:
            Dictionary containing preprocessing results and insights
        """
        logging.info("Starting comprehensive dataset preprocessing...")
        
        # Step 1: Data validation and basic analysis
        validation_results = self._validate_and_analyze_data(data)
        
        # Step 2: Intelligent feature analysis
        feature_analysis = self._perform_intelligent_feature_analysis(data, target_column, problem_type)
        
        # Step 3: Create preprocessing pipeline
        preprocessing_pipeline = self._create_intelligent_preprocessing_pipeline(data, target_column, problem_type)
        
        # Step 4: Apply preprocessing
        preprocessed_data = self._apply_preprocessing(data, preprocessing_pipeline)
        
        # Step 5: Generate intelligent recommendations
        recommendations = self._generate_preprocessing_recommendations(
            validation_results, feature_analysis, preprocessing_pipeline
        )
        
        # Combine all results
        preprocessing_results = {
            "original_data_shape": data.shape,
            "preprocessed_data_shape": preprocessed_data.shape if preprocessed_data is not None else None,
            "validation_results": validation_results,
            "feature_analysis": feature_analysis,
            "preprocessing_pipeline": preprocessing_pipeline,
            "preprocessed_data": preprocessed_data,
            "recommendations": recommendations
        }
        
        logging.info("Comprehensive dataset preprocessing completed successfully")
        return preprocessing_results
    
    def _validate_and_analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate and analyze the input data."""
        validation_results = {
            "data_types": data.dtypes.apply(lambda x: x.name).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "numerical_features": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_features": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "high_cardinality_features": [],
            "outlier_columns": []
        }
        
        # Detect high cardinality categorical features
        for col in validation_results["categorical_features"]:
            unique_count = data[col].nunique()
            if unique_count > 50:  # Threshold for high cardinality
                validation_results["high_cardinality_features"].append({
                    "column": col,
                    "unique_count": unique_count
                })
        
        # Detect outlier columns
        for col in validation_results["numerical_features"]:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if len(outliers) > len(data) * 0.05:  # More than 5% outliers
                validation_results["outlier_columns"].append({
                    "column": col,
                    "outlier_count": len(outliers),
                    "outlier_percentage": len(outliers) / len(data) * 100
                })
        
        return validation_results
    
    def _perform_intelligent_feature_analysis(self, data: pd.DataFrame, target_column: Optional[str], 
                                            problem_type: Optional[str]) -> Dict[str, Any]:
        """Perform intelligent feature analysis if target column is available."""
        if not target_column or not problem_type or target_column not in data.columns:
            logging.info("Skipping intelligent feature analysis - target column or problem type not available")
            return {"status": "skipped", "reason": "No target column or problem type provided"}
        
        try:
            # Prepare data for analysis
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Initialize feature analyzer
            feature_analyzer = IntelligentFeatureAnalyzer(target_column, problem_type)
            
            # Perform analysis
            feature_insights = feature_analyzer.analyze_features(X, y)
            
            logging.info("Intelligent feature analysis completed")
            return {
                "status": "completed",
                "insights": feature_insights,
                "summary": feature_insights.get("summary", "Analysis completed successfully")
            }
            
        except Exception as e:
            logging.error(f"Intelligent feature analysis failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _create_intelligent_preprocessing_pipeline(self, data: pd.DataFrame, target_column: Optional[str], 
                                                  problem_type: Optional[str]) -> Dict[str, Any]:
        """Create intelligent preprocessing pipeline based on data characteristics."""
        logging.info("Creating intelligent preprocessing pipeline...")
        
        # Get data summary
        data_summary = create_data_summary(data)
        
        # Get feature types
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from features if present
        if target_column and target_column in numerical_features:
            numerical_features.remove(target_column)
        elif target_column and target_column in categorical_features:
            categorical_features.remove(target_column)
        
        # Use ToolDecider for intelligent strategy selection
        tool_decider = get_tool_decider("rule_based")
        available_tools = ["imputation", "scaling", "encoding", "normalization"]
        decision = tool_decider.decide_preprocessing_strategy(data_summary, available_tools)
        
        logging.info(f"ToolDecider chose preprocessing strategy: {decision}")
        
        # Build numerical transformer
        numerical_steps = []
        if "imputation" in decision.get("tools", []):
            if data_summary["missing_percentage"] > 20:
                numerical_steps.append(('imputer', KNNImputer(n_neighbors=3)))
                logging.info("Using KNN imputation for high missing percentage")
            else:
                numerical_steps.append(('imputer', SimpleImputer(strategy='median')))
                logging.info("Using median imputation for low missing percentage")
        
        if "scaling" in decision.get("tools", []):
            if data_summary["memory_usage_mb"] > 100:  # Large dataset
                numerical_steps.append(('scaler', RobustScaler()))
                logging.info("Using RobustScaler for large dataset")
            else:
                numerical_steps.append(('scaler', StandardScaler()))
                logging.info("Using StandardScaler for standard preprocessing")
        elif "normalization" in decision.get("tools", []):
            numerical_steps.append(('scaler', MinMaxScaler()))
            logging.info("Using MinMaxScaler for normalization")
        
        # Build categorical transformer
        categorical_steps = []
        if "imputation" in decision.get("tools", []):
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        if "encoding" in decision.get("tools", []):
            # Filter out high cardinality features
            safe_categorical_features = []
            for feature in categorical_features:
                unique_count = data[feature].nunique()
                if unique_count <= 50:  # Safe for one-hot encoding
                    safe_categorical_features.append(feature)
                else:
                    logging.warning(f"High cardinality feature '{feature}' ({unique_count} unique values) - will be dropped")
            
            if safe_categorical_features:
                categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50)))
        
        # Create transformers
        numerical_transformer = Pipeline(steps=numerical_steps) if numerical_steps else None
        categorical_transformer = Pipeline(steps=categorical_steps) if categorical_steps else None
        
        # Create column transformer
        transformers = []
        if numerical_transformer and numerical_features:
            transformers.append(('num', numerical_transformer, numerical_features))
        if categorical_transformer and categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if transformers:
            column_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough')
        else:
            column_transformer = None
        
        pipeline_info = {
            "strategy": decision,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "numerical_steps": [step[0] for step in numerical_steps],
            "categorical_steps": [step[0] for step in categorical_steps],
            "column_transformer": column_transformer,
            "data_summary": data_summary
        }
        
        return pipeline_info
    
    def _apply_preprocessing(self, data: pd.DataFrame, pipeline_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Apply the preprocessing pipeline to the data."""
        try:
            column_transformer = pipeline_info.get("column_transformer")
            if column_transformer is None:
                logging.warning("No preprocessing pipeline to apply")
                return data.copy()
            
            # Apply preprocessing
            preprocessed_data = column_transformer.fit_transform(data)
            
            # Convert back to DataFrame
            feature_names = []
            for name, transformer, columns in column_transformer.transformers_:
                if name == 'num':
                    feature_names.extend(columns)
                elif name == 'cat':
                    # Get feature names from one-hot encoder
                    if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                        cat_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(columns)
                elif name == 'remainder':
                    feature_names.extend(columns)
            
            preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names, index=data.index)
            
            logging.info(f"Preprocessing applied successfully. Shape: {preprocessed_df.shape}")
            return preprocessed_df
            
        except Exception as e:
            logging.error(f"Failed to apply preprocessing: {e}")
            return None
    
    def _generate_preprocessing_recommendations(self, validation_results: Dict, feature_analysis: Dict, 
                                             pipeline_info: Dict) -> List[str]:
        """Generate intelligent preprocessing recommendations."""
        recommendations = []
        
        # Data quality recommendations
        if validation_results["missing_percentage"].sum() > 0:
            recommendations.append("Consider imputation strategies for missing values")
        
        if validation_results["duplicate_rows"] > 0:
            recommendations.append("Investigate and handle duplicate records")
        
        # High cardinality recommendations
        if validation_results["high_cardinality_features"]:
            recommendations.append("Consider target encoding or feature hashing for high cardinality categorical features")
        
        # Outlier recommendations
        if validation_results["outlier_columns"]:
            recommendations.append("Apply robust scaling or outlier detection for columns with significant outliers")
        
        # Feature analysis recommendations
        if feature_analysis.get("status") == "completed":
            insights = feature_analysis.get("insights", {})
            if insights.get("high_correlation_features"):
                recommendations.append("Consider removing highly correlated features to reduce multicollinearity")
            
            if insights.get("low_importance_features"):
                recommendations.append("Consider removing low-importance features to improve model performance")
        
        # Manufacturing-specific recommendations
        sensor_features = [col for col in validation_results["numerical_features"] if any(
            keyword in col.lower() for keyword in ['temp', 'temperature', 'pressure', 'vibration', 'acoustic']
        )]
        if sensor_features:
            recommendations.append("Apply sensor-specific preprocessing (noise reduction, outlier handling)")
        
        # Try to get Gemini insights if available
        if self.llm is not None:
            try:
                gemini_recommendations = self._get_gemini_preprocessing_insights(
                    validation_results, feature_analysis, pipeline_info
                )
                recommendations.extend(gemini_recommendations)
            except Exception as e:
                logging.warning(f"Failed to get Gemini preprocessing insights: {e}")
        
        return recommendations
    
    def _get_gemini_preprocessing_insights(self, validation_results: Dict, feature_analysis: Dict, 
                                        pipeline_info: Dict) -> List[str]:
        """Get intelligent preprocessing insights from Gemini."""
        try:
            summary = f"""
            Manufacturing Dataset Preprocessing Analysis:
            - Original Shape: {validation_results.get('original_shape', 'Unknown')}
            - Missing Values: {sum(validation_results['missing_percentage'].values()):.1f}% total
            - Categorical Features: {len(validation_results['categorical_features'])}
            - Numerical Features: {len(validation_results['numerical_features'])}
            - High Cardinality Features: {len(validation_results['high_cardinality_features'])}
            - Outlier Columns: {len(validation_results['outlier_columns'])}
            - Preprocessing Strategy: {pipeline_info.get('strategy', {}).get('tools', [])}
            """
            
            prompt = f"""
            As a manufacturing data preprocessing expert, analyze this preprocessing plan and provide 3-5 specific, actionable recommendations:
            
            {summary}
            
            Focus on:
            1. Manufacturing-specific preprocessing strategies
            2. Sensor data handling techniques
            3. Feature engineering opportunities
            4. Data quality improvements
            5. Model performance optimization
            
            Provide concise, actionable recommendations for manufacturing data preprocessing.
            """
            
            response = self.llm.invoke(prompt)
            gemini_recommendations = [line.strip() for line in response.content.split('\n') if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
            
            return gemini_recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logging.warning(f"Failed to get Gemini preprocessing insights: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Test the agent
    agent = PreprocessingIntelligenceAgent()
    
    # Test with sample data
    import pandas as pd
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })
    
    results = agent.preprocess_dataset(sample_data, target_column='target', problem_type='classification')
    print("Preprocessing test completed successfully!")
    print(f"Original shape: {results['original_data_shape']}")
    print(f"Preprocessed shape: {results['preprocessed_data_shape']}")
    print(f"Recommendations: {len(results['recommendations'])} generated")
