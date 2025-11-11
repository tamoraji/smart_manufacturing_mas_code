
import pandas as pd
import logging
from typing import Dict, Any
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

    def _infer_target_column(self, data: pd.DataFrame, predictions) -> str:
        """Attempt to infer the target column by matching prediction values."""
        try:
            prediction_strings = set(pd.Series(predictions).astype(str).unique())
        except Exception:
            prediction_strings = set(str(p) for p in set(predictions))

        for column in data.columns:
            series = data[column]
            try:
                values = set(series.astype(str).unique())
            except Exception:
                values = set(series.apply(str).unique())
            if prediction_strings.issubset(values):
                return column
        return ""

    @staticmethod
    def _is_numeric_label(label) -> bool:
        if isinstance(label, (int, float, np.number)):
            return True
        if isinstance(label, str):
            try:
                float(label)
                return True
            except ValueError:
                return False
        return False

    def _score_from_keywords(self, label: str, target_column: str) -> float:
        """Derive a priority score (1-3) based on label semantics and context."""
        label_lower = str(label).strip().lower()
        if not label_lower:
            return 2.0

        target_lower = target_column.lower() if target_column else ""

        # Context where lower textual value implies higher risk (e.g., efficiency, health)
        invert_low_high = any(
            keyword in target_lower
            for keyword in ["efficiency", "performance", "quality", "health", "uptime", "score", "yield"]
        )

        # Priority-specific cues override general rules
        if "priority" in label_lower or "risk" in label_lower or "severity" in label_lower:
            if "high" in label_lower:
                return 3.0
            if "medium" in label_lower:
                return 2.0
            if "low" in label_lower:
                return 1.0

        high_severity_keywords = [
            "critical", "urgent", "fail", "failure", "fault", "alarm",
            "down", "offline", "shutdown", "alert", "incident", "unsafe",
            "issue", "anomaly", "breach", "hazard", "poor", "degraded", "risk"
        ]
        medium_severity_keywords = [
            "medium", "moderate", "warning", "reduced", "caution",
            "watch", "elevated", "unstable", "attention"
        ]
        low_severity_keywords = [
            "normal", "ok", "good", "optimal", "stable",
            "healthy", "excellent", "nominal", "efficient"
        ]

        # Handle generic "high"/"low" depending on context
        if invert_low_high:
            high_severity_keywords.append("low")
            low_severity_keywords.append("high")
        else:
            high_severity_keywords.append("high")
            low_severity_keywords.append("low")

        for keyword in high_severity_keywords:
            if keyword in label_lower:
                return 3.0
        for keyword in medium_severity_keywords:
            if keyword in label_lower:
                return 2.0
        for keyword in low_severity_keywords:
            if keyword in label_lower:
                return 1.0

        # Default when semantics are unclear
        return 2.0

    def _derive_priority_mapping(self, predictions, target_column: str) -> Dict[Any, float]:
        """Create a mapping from prediction labels to numeric priority scores."""
        unique_seen = []
        for label in predictions:
            if label not in unique_seen:
                unique_seen.append(label)

        numeric_labels = []
        mapping: Dict[Any, float] = {}

        for label in unique_seen:
            if self._is_numeric_label(label):
                try:
                    numeric_value = float(label)
                except ValueError:
                    numeric_value = float(str(label))
                numeric_labels.append((label, numeric_value))
            else:
                mapping[label] = self._score_from_keywords(label, target_column)

        if numeric_labels:
            numeric_values = [value for _, value in numeric_labels]
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            for original_label, numeric_value in numeric_labels:
                if max_val == min_val:
                    score = 2.0
                else:
                    score = 1.0 + 2.0 * ((numeric_value - min_val) / (max_val - min_val))
                mapping[original_label] = float(score)

        return mapping

    def _build_contributing_factors(self, row: pd.Series, feature_names: list[str]) -> str:
        factors = []
        for feature in feature_names:
            if feature not in row:
                continue
            value = row[feature]
            if isinstance(value, (int, float, np.number)):
                factors.append(f"{feature}={value:.2f}")
            else:
                factors.append(f"{feature}={value}")
        return ", ".join(factors)

    def _action_plan_from_score(self, score: float) -> Dict[str, str]:
        if score >= 2.5:
            return {
                "priority_level": "Critical",
                "action": "IMMEDIATE: Dispatch maintenance team to investigate and recover performance.",
                "cost": "High ($5,000+)",
                "timeframe": "Within 24-48 hours"
            }
        if score >= 1.5:
            return {
                "priority_level": "Elevated",
                "action": "Schedule targeted maintenance in the upcoming service window.",
                "cost": "Medium ($1,000-$5,000)",
                "timeframe": "Within 1-2 weeks"
            }
        return {
            "priority_level": "Low",
            "action": "Continue monitoring and maintain current operating procedures.",
            "cost": "Low (<$1,000)",
            "timeframe": "Next scheduled maintenance"
        }

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
            results_df = self.results['results_df'].copy()

            if not results_df.empty:
                identifier_cols = [col for col in results_df.columns if col.startswith("identifier__")]
                if identifier_cols:
                    for id_col in identifier_cols:
                        results_df[id_col.replace("identifier__", "")] = results_df[id_col]

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
            # Handle classification results with enhanced, label-aware insights
            results_df = self.results['test_data'].copy()
            predictions = pd.Series(self.results['test_predictions'], index=results_df.index, name="Predicted_Label")
            results_df['Predicted_Label'] = predictions

            target_column = self._infer_target_column(results_df, predictions)
            priority_mapping = self._derive_priority_mapping(predictions, target_column)

            default_score = 2.0 if priority_mapping else 2.0
            results_df['Priority_Score'] = results_df['Predicted_Label'].map(priority_mapping).fillna(default_score)

            # Prepare feature context
            feature_names = []
            if (
                self.results.get('feature_importances') is not None
                and not self.results['feature_importances'].empty
                and 'feature' in self.results['feature_importances']
            ):
                feature_names = [
                    f.replace('num__', '').replace('cat__', '')
                    for f in self.results['feature_importances']['feature'].head(3).tolist()
                ]
            elif 'test_data' in self.results:
                feature_names = [
                    col for col in self.results['test_data'].columns
                    if col not in {'Machine_ID', 'Timestamp'}
                ][:3]

            prioritized = results_df.sort_values('Priority_Score', ascending=False)
            if 'Machine_ID' in prioritized.columns:
                prioritized = prioritized.dropna(subset=['Machine_ID'])
                prioritized = prioritized.drop_duplicates(subset=['Machine_ID'], keep='first')

            max_recommendations = 30
            prioritized = prioritized.head(max_recommendations)

            recommendations = []
            for _, row in prioritized.iterrows():
                score = float(row.get('Priority_Score', default_score))
                label_text = str(row.get('Predicted_Label', 'Unknown'))
                action_plan = self._action_plan_from_score(score)
                contributing = self._build_contributing_factors(row, feature_names)

                reason = f"Model predicted '{label_text}' for the current operating state."
                if feature_names:
                    readable_features = ", ".join(feature_names)
                    reason += f" Key drivers include: {readable_features}."

                recommendations.append({
                    'Machine_ID': row.get('Machine_ID', 'Unknown'),
                    'Predicted_Label': label_text,
                    'Priority_Level': action_plan['priority_level'],
                    'Priority_Score': round(score, 2),
                    'Contributing_Factors': contributing if contributing else "Model-driven signals (top features unavailable).",
                    'Reason_for_Action': reason,
                    'Recommended_Action': action_plan['action'],
                    'Estimated_Cost': action_plan['cost'],
                    'Timeframe': action_plan['timeframe'],
                    'Model_Confidence': model_performance['recommendation_confidence']
                })

            if recommendations:
                recommendations_df = pd.DataFrame(recommendations).sort_values('Priority_Score', ascending=False)

                if model_performance['reliability_warning']:
                    logging.warning(f"⚠️ {model_performance['reliability_warning']}")
                    recommendations_df['Model_Warning'] = model_performance['reliability_warning']

                return recommendations_df

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
