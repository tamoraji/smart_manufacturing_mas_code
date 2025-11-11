
import pandas as pd
import logging
from typing import Dict, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class OptimizationAgent:
    """
    The OptimizationAgent takes insights from the AnalysisAgent and generates
    a prescriptive maintenance action plan. It represents the final step in
    closing the loop from prediction to action.
    """
    def __init__(self, analysis_results: Dict[str, Any]):
        """
        Initialize the OptimizationAgent.
        Args:
            analysis_results (dict): A dictionary containing the results from the
                                     AnalysisAgent, including predictions, original test data
                                     for context, and feature importances.
        """
        logging.info("Initializing Optimization Agent...")
        self.results = analysis_results
        # Check for required keys based on analysis type
        if 'results_df' in analysis_results:
            # Anomaly detection results
            if 'results_df' not in analysis_results or 'anomaly_labels' not in analysis_results:
                raise ValueError("Anomaly detection results missing required keys: 'results_df' and 'anomaly_labels'")
        else:
            # Supervised learning results - train_predictions only needed for regression
            required_keys = ['test_data', 'test_predictions', 'feature_importances']
            if not all(k in self.results for k in required_keys):
                raise ValueError("Analysis results are missing required keys for optimization.")

    def _assess_model_performance(self) -> Dict[str, Any]:
        """Assess model performance and provide context for recommendations."""
        performance = {
            'confidence_level': 'High',
            'reliability_warning': None,
            'recommendation_confidence': 'High'
        }
        
        # Check for common performance issues
        if 'accuracy' in self.results:
            accuracy = self.results['accuracy']
            if accuracy < 0.6:
                performance['confidence_level'] = 'Low'
                performance['reliability_warning'] = f"Model accuracy ({accuracy:.2%}) is below recommended threshold (60%)"
                performance['recommendation_confidence'] = 'Low'
            elif accuracy < 0.8:
                performance['confidence_level'] = 'Medium'
                performance['reliability_warning'] = f"Model accuracy ({accuracy:.2%}) suggests moderate reliability"
                performance['recommendation_confidence'] = 'Medium'
        
        if 'r2' in self.results:
            r2 = self.results['r2']
            if r2 < 0:
                performance['confidence_level'] = 'Very Low'
                performance['reliability_warning'] = f"Model performs worse than baseline (R² = {r2:.3f})"
                performance['recommendation_confidence'] = 'Very Low'
            elif r2 < 0.3:
                performance['confidence_level'] = 'Low'
                performance['reliability_warning'] = f"Model explains only {r2:.1%} of variance"
                performance['recommendation_confidence'] = 'Low'
        
        return performance

    def generate_recommendations(self) -> pd.DataFrame:
        """
        Generates comprehensive recommendations based on analysis results.
        For classification: Prioritizes maintenance tasks with detailed insights
        For regression: Identifies concerning predicted values with context
        For anomaly detection: Identifies and explains anomalous behavior
        Returns:
            pd.DataFrame: A DataFrame containing recommendations and actions
        """
        logging.info("Generating comprehensive prescriptive recommendations...")
        
        # Add model performance context to recommendations
        model_performance = self._assess_model_performance()
        logging.info(f"Model performance assessment: {model_performance}")
        
        train_predictions = self.results.get('train_predictions')
        is_regression = (
            self.results.get('r2') is not None and
            train_predictions is not None and
            np.issubdtype(np.asarray(train_predictions).dtype, np.number)
        )

        if is_regression:
            # Handle regression results
            results_df = self.results['test_data'].copy()
            results_df['Predicted_Value'] = self.results['test_predictions']
            
            # Calculate prediction thresholds based on training data distribution
            train_mean = np.asarray(train_predictions).mean()
            train_std = np.asarray(train_predictions).std()
            high_threshold = train_mean + 2 * train_std
            critical_threshold = train_mean + 3 * train_std
            
            # Identify machines with concerning predicted values
            critical = results_df[results_df['Predicted_Value'] >= critical_threshold].copy()
            warning = results_df[(results_df['Predicted_Value'] >= high_threshold) & 
                               (results_df['Predicted_Value'] < critical_threshold)].copy()
            
            recommendations = []
            
            # Handle critical cases
            for _, row in critical.iterrows():
                recommendations.append({
                    'Machine_ID': row['Machine_ID'] if 'Machine_ID' in row else 'Unknown',
                    'Priority_Score': (row['Predicted_Value'] - train_mean) / train_std,
                    'Current_Value': row['Predicted_Value'],
                    'Threshold': critical_threshold,
                    'Severity': 'Critical',
                    'Reason_for_Action': f"Predicted value ({row['Predicted_Value']:.2f}) exceeds critical threshold ({critical_threshold:.2f})",
                    'Recommended_Action': "Immediate inspection and preventive maintenance required"
                })
            
            # Handle warning cases
            for _, row in warning.iterrows():
                recommendations.append({
                    'Machine_ID': row['Machine_ID'] if 'Machine_ID' in row else 'Unknown',
                    'Priority_Score': (row['Predicted_Value'] - train_mean) / train_std,
                    'Current_Value': row['Predicted_Value'],
                    'Threshold': high_threshold,
                    'Severity': 'Warning',
                    'Reason_for_Action': f"Predicted value ({row['Predicted_Value']:.2f}) exceeds warning threshold ({high_threshold:.2f})",
                    'Recommended_Action': "Schedule inspection within next maintenance window"
                })
            
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                return recommendations_df.sort_values('Priority_Score', ascending=False)
            else:
                logging.info("No concerning predictions identified. All values within normal range.")
                return pd.DataFrame()
                
        elif 'anomaly_labels' in self.results:
            # Handle anomaly detection results
            results_df = self.results['results_df']
            anomalous = results_df[results_df['Is_Anomaly']]
            
            if anomalous.empty:
                logging.info("No anomalies detected. All machines operating within normal parameters.")
                return pd.DataFrame()
            
            # Group anomalies by Machine_ID
            machine_anomalies = anomalous.groupby('Machine_ID').agg({
                'Is_Anomaly': 'count',
                'Anomaly_Score': 'mean'
            }).reset_index()
            
            machine_anomalies.columns = ['Machine_ID', 'Anomaly_Count', 'Avg_Anomaly_Score']
            machine_anomalies = machine_anomalies.sort_values('Avg_Anomaly_Score')
            
            # Generate recommendations for each anomalous machine
            recommendations = []
            for _, row in machine_anomalies.iterrows():
                machine_data = anomalous[anomalous['Machine_ID'] == row['Machine_ID']]
                
                # Find the most extreme deviations
                extreme_cols = []
                for col in machine_data.columns:
                    if col.endswith('_zscore'):
                        metric = col.replace('_zscore', '')
                        mean_zscore = machine_data[col].mean()
                        if abs(mean_zscore) > 2:  # More than 2 standard deviations
                            extreme_cols.append(f"{metric} ({mean_zscore:.1f}σ)")
                
                recommendations.append({
                    'Machine_ID': row['Machine_ID'],
                    'Priority_Score': abs(row['Avg_Anomaly_Score']),
                    'Anomaly_Count': row['Anomaly_Count'],
                    'Reason_for_Action': f"Anomalous behavior detected in: {', '.join(extreme_cols)}",
                    'Recommended_Action': "Schedule inspection and diagnostic testing" if len(extreme_cols) > 2 
                                       else "Monitor these parameters closely"
                })
            
            recommendations_df = pd.DataFrame(recommendations)
            
            # Add timestamp information
            if 'Timestamp' in results_df.columns:
                first_anomaly = results_df.groupby('Machine_ID')['Timestamp'].min().reset_index()
                first_anomaly.columns = ['Machine_ID', 'First_Anomaly_Time']
                recommendations_df = recommendations_df.merge(first_anomaly, on='Machine_ID')
            
            return recommendations_df
            
        else:
            # Handle classification results with enhanced insights
            recommendations_df = self.results['test_data'].copy()
            recommendations_df['Predicted_Priority'] = self.results['test_predictions']
            
            # Get top contributing features (with error handling)
            if self.results.get('feature_importances') is not None and 'feature' in self.results['feature_importances']:
                top_features = self.results['feature_importances']['feature'].head(3).tolist()
                feature_names = [f.replace('num__', '').replace('cat__', '') for f in top_features]
            else:
                # Fallback: use column names from the data
                feature_names = list(self.results['test_data'].columns)[:3] if 'test_data' in self.results else ['Feature1', 'Feature2', 'Feature3']
                logging.warning("Feature importances not available, using column names as fallback")
            
            # Generate recommendations for all priority levels
            recommendations = []
            
            # High priority (3) - Critical maintenance needed
            high_priority = recommendations_df[recommendations_df['Predicted_Priority'] == 3].copy()
            for _, row in high_priority.iterrows():
                # Analyze specific feature values contributing to high priority
                contributing_factors = []
                for feature in top_features:
                    clean_name = feature.replace('num__', '').replace('cat__', '')
                    if clean_name in row.index:
                        value = row[clean_name]
                        contributing_factors.append(f"{clean_name}: {value:.2f}")
                
                recommendations.append({
                    'Machine_ID': row.get('Machine_ID', 'Unknown'),
                    'Priority_Level': 'Critical (3)',
                    'Priority_Score': 3.0,
                    'Contributing_Factors': ', '.join(contributing_factors),
                    'Reason_for_Action': f"Critical maintenance predicted based on: {', '.join(feature_names)}",
                    'Recommended_Action': "IMMEDIATE: Schedule emergency inspection and prepare for component replacement",
                    'Estimated_Cost': "High ($5,000-$50,000)",
                    'Timeframe': "Within 24-48 hours",
                    'Model_Confidence': model_performance['recommendation_confidence']
                })
            
            # Medium priority (2) - Preventive maintenance
            medium_priority = recommendations_df[recommendations_df['Predicted_Priority'] == 2].copy()
            for _, row in medium_priority.iterrows():
                # Analyze specific feature values contributing to medium priority
                contributing_factors = []
                for feature in top_features:
                    clean_name = feature.replace('num__', '').replace('cat__', '')
                    if clean_name in row.index:
                        value = row[clean_name]
                        contributing_factors.append(f"{clean_name}: {value:.2f}")
                
                recommendations.append({
                    'Machine_ID': row.get('Machine_ID', 'Unknown'),
                    'Priority_Level': 'Medium (2)',
                    'Priority_Score': 2.0,
                    'Contributing_Factors': ', '.join(contributing_factors) if contributing_factors else "Moderate risk indicators detected",
                    'Reason_for_Action': f"Preventive maintenance recommended based on: {', '.join(feature_names)}",
                    'Recommended_Action': "Schedule routine inspection and minor maintenance",
                    'Estimated_Cost': "Medium ($500-$5,000)",
                    'Timeframe': "Within 1-2 weeks",
                    'Model_Confidence': model_performance['recommendation_confidence']
                })
            
            # Low priority (1) - Monitoring
            low_priority = recommendations_df[recommendations_df['Predicted_Priority'] == 1].copy()
            for _, row in low_priority.iterrows():
                # Analyze specific feature values contributing to low priority
                contributing_factors = []
                for feature in top_features:
                    clean_name = feature.replace('num__', '').replace('cat__', '')
                    if clean_name in row.index:
                        value = row[clean_name]
                        contributing_factors.append(f"{clean_name}: {value:.2f}")
                
                recommendations.append({
                    'Machine_ID': row.get('Machine_ID', 'Unknown'),
                    'Priority_Level': 'Low (1)',
                    'Priority_Score': 1.0,
                    'Contributing_Factors': ', '.join(contributing_factors) if contributing_factors else "Normal operating parameters",
                    'Reason_for_Action': f"Continue monitoring based on: {', '.join(feature_names)}",
                    'Recommended_Action': "Continue routine monitoring and scheduled maintenance",
                    'Estimated_Cost': "Low ($100-$500)",
                    'Timeframe': "Next scheduled maintenance",
                    'Model_Confidence': model_performance['recommendation_confidence']
                })
            
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                
                # Add model performance warning if needed
                if model_performance['reliability_warning']:
                    logging.warning(f"⚠️ {model_performance['reliability_warning']}")
                    recommendations_df['Model_Warning'] = model_performance['reliability_warning']
                
                return recommendations_df.sort_values('Priority_Score', ascending=False)
            else:
                logging.info("No maintenance recommendations generated.")
                return pd.DataFrame()

    def generate_summary_report(self, recommendations_df: pd.DataFrame) -> str:
        """Generate a human-readable summary report of recommendations."""
        if recommendations_df.empty:
            return "✅ All systems operating within normal parameters. No immediate maintenance actions required."
        
        report_lines = ["🎯 MAINTENANCE RECOMMENDATIONS SUMMARY", "=" * 50]
        
        # Count by priority
        if 'Priority_Level' in recommendations_df.columns:
            priority_counts = recommendations_df['Priority_Level'].value_counts()
            report_lines.append(f"\n📊 Priority Distribution:")
            for priority, count in priority_counts.items():
                report_lines.append(f"  • {priority}: {count} machines")
        
        # Cost estimation
        if 'Estimated_Cost' in recommendations_df.columns:
            cost_summary = recommendations_df['Estimated_Cost'].value_counts()
            report_lines.append(f"\n💰 Estimated Cost Distribution:")
            for cost, count in cost_summary.items():
                report_lines.append(f"  • {cost}: {count} machines")
        
        # Top recommendations
        top_recs = recommendations_df.head(3)
        report_lines.append(f"\n🚨 TOP PRIORITY ACTIONS:")
        for _, rec in top_recs.iterrows():
            report_lines.append(f"  • Machine {rec.get('Machine_ID', 'Unknown')}: {rec.get('Recommended_Action', 'N/A')}")
        
        # Model confidence warning
        if 'Model_Warning' in recommendations_df.columns and not recommendations_df['Model_Warning'].isna().all():
            warnings = recommendations_df['Model_Warning'].dropna().unique()
            if warnings:
                report_lines.append(f"\n⚠️ MODEL RELIABILITY WARNING:")
                for warning in warnings:
                    report_lines.append(f"  • {warning}")
        
        return "\n".join(report_lines)

if __name__ == '__main__':
    logging.info("--- Running Optimization Agent in Standalone Mode ---")
    
    # Create sample analysis results for demonstration
    sample_analysis_results = {
        'test_data': pd.DataFrame({
            'Machine_ID': ['M01', 'M02', 'M03', 'M04'],
            'Temp_C': [308.1, 300.5, 309.2, 299.8],
            'Vibration_mm_s': [2.5, 1.2, 2.8, 1.1]
        }),
        'test_predictions': [3, 1, 3, 2], # Two high-priority predictions
        'feature_importances': pd.DataFrame({
            'feature': ['num__Vibration_mm_s', 'num__Temp_C'],
            'importance': [0.6, 0.4]
        })
    }
    
    # Initialize and run the agent
    optimization_agent = OptimizationAgent(sample_analysis_results)
    optimization_agent.generate_recommendations()
    
    logging.info("--- End of Standalone Run ---")
