"""
Enhanced Reporting System for Intelligent MAS
Provides comprehensive reporting and visualization of workflow results.
"""

import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class WorkflowReporter:
    """Enhanced reporting system for workflow results and analysis."""
    
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        self.report_data = {
            'workflow_start': datetime.now().isoformat(),
            'workflow_goal': None,
            'steps': [],
            'performance_metrics': {},
            'feature_analysis': {},
            'recommendations': {},
            'stage_summaries': [],
            'hitl_events': [],
            'errors': [],
            'summary': {}
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def log_step(self, step_name: str, success: bool, message: str, 
                 details: Optional[Dict[str, Any]] = None, duration: Optional[float] = None):
        """Log a workflow step with enhanced details."""
        step_data = {
            'step_name': step_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'details': details or {}
        }
        
        self.report_data['steps'].append(step_data)
        
        # Enhanced console output
        status_icon = "✅" if success else "❌"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        
        logging.info(f"{status_icon} {step_name}: {message}{duration_str}")
        
        if details:
            for key, value in details.items():
                if isinstance(value, dict):
                    logging.info(f"   📊 {key}: {json.dumps(value, indent=2)}")
                else:
                    logging.info(f"   📊 {key}: {value}")
    
    def log_stage_summary(self, stage_name: str, summary: str, stats: Optional[Dict[str, Any]] = None):
        """
        Log a structured stage summary with optional bullet-point statistics.
        """
        stats = stats or {}
        self.report_data['stage_summaries'].append({
            'stage_name': stage_name,
            'summary': summary,
            'stats': stats
        })

        logging.info("")
        logging.info(f"--- {stage_name} ---")
        logging.info(summary)
        for key, value in stats.items():
            logging.info(f"   • {key}: {value}")
    
    def log_hitl_event(self, title: str, decision: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a human-in-the-loop interaction with explicit console output and persistence.
        """
        metadata = metadata or {}
        event_entry = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "decision": decision,
            "metadata": metadata
        }
        self.report_data['hitl_events'].append(event_entry)

        context_bits = ", ".join(f"{k}={v}" for k, v in metadata.items()) if metadata else ""
        suffix = f" ({context_bits})" if context_bits else ""
        logging.info(f"👤 HITL · {title}: {decision}{suffix}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log model performance metrics with enhanced formatting."""
        self.report_data['performance_metrics'] = metrics
        
        logging.info("🎯 PERFORMANCE METRICS")
        logging.info("=" * 50)
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                if 'accuracy' in metric_name.lower() or 'r2' in metric_name.lower():
                    logging.info(f"   {metric_name}: {value:.4f} ({value*100:.2f}%)")
                else:
                    logging.info(f"   {metric_name}: {value:.4f}")
            else:
                logging.info(f"   {metric_name}: {value}")
    
    def log_feature_analysis(self, analysis: Dict[str, Any]):
        """Log intelligent feature analysis results."""
        self.report_data['feature_analysis'] = analysis
        
        logging.info("🧠 INTELLIGENT FEATURE ANALYSIS")
        logging.info("=" * 50)
        
        try:
            if 'summary' in analysis:
                logging.info(analysis['summary'])
            
            if 'recommendations' in analysis:
                recs = analysis['recommendations']
                
                if 'features_to_remove' in recs and recs['features_to_remove']:
                    logging.info("\n🗑️ Features Removed:")
                    for feature in recs['features_to_remove']:
                        feature_name = feature.get('feature', 'Unknown')
                        reason = feature.get('reason', 'No reason provided')
                        logging.info(f"   • {feature_name}: {reason}")
                
                if 'features_to_keep' in recs and recs['features_to_keep']:
                    logging.info("\n⭐ Top Features to Keep:")
                    for feature in recs['features_to_keep'][:5]:  # Show top 5
                        feature_name = feature.get('feature', 'Unknown')
                        score = feature.get('score', 0.0)
                        logging.info(f"   • {feature_name}: {score:.3f}")
        except Exception as e:
            logging.warning(f"Error logging feature analysis: {e}")
            logging.info("Feature analysis completed but detailed logging failed.")
    
    def log_recommendations(self, recommendations: Dict[str, Any]):
        """Log prescriptive recommendations with enhanced formatting."""
        self.report_data['recommendations'] = recommendations
        
        logging.info("🎯 PRESCRIPTIVE RECOMMENDATIONS")
        logging.info("=" * 50)
        
        if 'summary_report' in recommendations:
            logging.info(recommendations['summary_report'])
        
        if 'recommendations' in recommendations:
            recs = recommendations['recommendations']
            # Handle both DataFrame and list formats
            if hasattr(recs, 'empty'):  # It's a DataFrame
                if not recs.empty:
                    logging.info("\n📋 Detailed Recommendations:")
                    for i, (_, rec) in enumerate(recs.iterrows(), 1):
                        self._log_single_recommendation(i, rec)
            elif recs:  # It's a list or other iterable
                logging.info("\n📋 Detailed Recommendations:")
                for i, rec in enumerate(recs, 1):
                    self._log_single_recommendation(i, rec)
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Log errors with enhanced context."""
        error_data = {
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        self.report_data['errors'].append(error_data)
        
        logging.error(f"❌ {error_type}: {error_message}")
        if context:
            for key, value in context.items():
                logging.error(f"   Context: {key} = {value}")
    
    def save_publication_snapshot(self, filename_prefix: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Save a compact, publication-ready snapshot of the workflow including stage summaries,
        key metrics, and a trimmed recommendation table.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix or f"publication_snapshot_{timestamp}"
        json_path = os.path.join(self.output_dir, f"{prefix}.json")
        csv_path = None

        snapshot = {
            "generated_at": datetime.now().isoformat(),
            "stage_summaries": self.report_data.get('stage_summaries', []),
            "performance_metrics": self.report_data.get('performance_metrics', {}),
            "workflow_summary": self.report_data.get('summary', {})
        }

        recommendations_obj = self.report_data.get('recommendations', {})
        recs = recommendations_obj.get('recommendations')
        if hasattr(recs, 'empty') and not recs.empty:
            # Choose informative columns if available
            preferred_columns = [
                'Machine_ID', 'Priority_Level', 'Recommended_Action',
                'Top_Indicators', 'Anomaly_Count', 'Avg_Anomaly_Score',
                'First_Anomaly_Time', 'Last_Anomaly_Time', 'Model_Confidence'
            ]
            available_cols = [col for col in preferred_columns if col in recs.columns]
            trimmed_recs = recs[available_cols].copy() if available_cols else recs.copy()
            snapshot['recommendations'] = trimmed_recs.head(25).to_dict(orient='records')

            csv_path = os.path.join(self.output_dir, f"{prefix}_recommendations.csv")
            trimmed_recs.to_csv(csv_path, index=False)
        elif isinstance(recs, list) and recs:
            snapshot['recommendations'] = recs[:25]
        else:
            snapshot['recommendations'] = []

        with open(json_path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

        logging.info(f"📄 Publication snapshot saved to: {json_path}")
        if csv_path:
            logging.info(f"📊 Recommendation preview saved to: {csv_path}")

        return {"json": json_path, "csv": csv_path}

    def _log_single_recommendation(self, index: int, rec: Dict[str, Any]):
        """Format a single recommendation line, skipping empty placeholders."""
        def _maybe_value(*keys):
            for key in keys:
                if key in rec and rec[key] is not None:
                    val = rec[key]
                    if isinstance(val, float) and pd.isna(val):
                        continue
                    return val
            return None

        action_value = _maybe_value('Action', 'Recommended_Action')
        machine_id = _maybe_value('Machine_ID')
        if action_value and machine_id and str(action_value).strip().lower() not in {'n/a', 'na'}:
            action_text = f"Machine {machine_id}: {action_value}"
        elif action_value:
            action_text = str(action_value)
        elif machine_id:
            action_text = f"Machine {machine_id}"
        else:
            action_text = "Action unavailable"

        logging.info(f"\n   {index}. {action_text}")

        priority = _maybe_value('Priority_Level', 'Priority')
        if priority and str(priority).strip().lower() not in {'n/a', 'unknown', 'none', ''}:
            logging.info(f"      Priority: {priority}")

        confidence = _maybe_value('Model_Confidence', 'Confidence')
        if confidence and str(confidence).strip().lower() not in {'n/a', 'unknown', 'none', ''}:
            logging.info(f"      Confidence: {confidence}")

        if 'Anomaly_Count' in rec and not pd.isna(rec['Anomaly_Count']):
            anomaly_info = f"{int(rec['Anomaly_Count'])} instances"
            avg_score = rec.get('Avg_Anomaly_Score')
            if avg_score is not None and not pd.isna(avg_score):
                anomaly_info += f", avg score {avg_score:.4f}"
            worst_score = rec.get('Most_Anomalous_Score')
            if worst_score is not None and not pd.isna(worst_score):
                anomaly_info += f", most anomalous {worst_score:.4f}"
            logging.info(f"      Anomaly Summary: {anomaly_info}")

        factors = _maybe_value('Contributing_Factors')
        if not factors:
            factors = _maybe_value('Top_Indicators')
        if factors and str(factors).strip():
            logging.info(f"      Factors: {factors}")

        cost = _maybe_value('Estimated_Cost')
        if cost and str(cost).strip().lower() not in {'n/a', 'unknown', 'none', ''}:
            logging.info(f"      Cost: {cost}")

        timeframe = _maybe_value('Timeframe')
        if timeframe and str(timeframe).strip().lower() not in {'n/a', 'unknown', 'none', ''}:
            logging.info(f"      Timeframe: {timeframe}")

        reason = _maybe_value('Reason_for_Action')
        if reason and str(reason).strip():
            logging.info(f"      Reason: {reason}")

    def generate_summary(self):
        """Generate a comprehensive workflow summary."""
        total_steps = len(self.report_data['steps'])
        successful_steps = sum(1 for step in self.report_data['steps'] if step['success'])
        failed_steps = total_steps - successful_steps
        
        # Calculate total duration
        start_time = datetime.fromisoformat(self.report_data['workflow_start'])
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Performance summary
        performance_summary = {}
        if self.report_data['performance_metrics']:
            metrics = self.report_data['performance_metrics']
            if 'r2' in metrics:
                performance_summary['r2_score'] = metrics['r2']
            if 'accuracy' in metrics:
                performance_summary['accuracy'] = metrics['accuracy']
            if 'mse' in metrics:
                performance_summary['mse'] = metrics['mse']
        
        # Feature analysis summary
        feature_summary = {}
        if self.report_data['feature_analysis']:
            analysis = self.report_data['feature_analysis']
            if 'recommendations' in analysis:
                recs = analysis['recommendations']
                feature_summary['features_removed'] = len(recs.get('features_to_remove', []))
                feature_summary['features_kept'] = len(recs.get('features_to_keep', []))
        
        # Recommendations summary
        rec_summary = {}
        if self.report_data['recommendations']:
            recs = self.report_data['recommendations']
            if 'recommendations' in recs:
                recommendations_list = recs['recommendations']
                # Handle both DataFrame and list formats
                if hasattr(recommendations_list, 'iterrows'):  # It's a DataFrame
                    rec_summary['total_recommendations'] = len(recommendations_list)
                    priority_counts = {}
                    for _, row in recommendations_list.iterrows():
                        priority = row.get('Priority_Level', 'Unknown')
                        priority_counts[priority] = priority_counts.get(priority, 0) + 1
                    rec_summary['priority_distribution'] = priority_counts
                elif isinstance(recommendations_list, list):  # It's a list
                    rec_summary['total_recommendations'] = len(recommendations_list)
                    priority_counts = {}
                    for rec in recommendations_list:
                        priority = rec.get('Priority_Level', 'Unknown')
                        priority_counts[priority] = priority_counts.get(priority, 0) + 1
                    rec_summary['priority_distribution'] = priority_counts
        
        self.report_data['summary'] = {
            'workflow_duration_seconds': total_duration,
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'performance_metrics': performance_summary,
            'feature_analysis': feature_summary,
            'recommendations': rec_summary,
            'workflow_end': end_time.isoformat()
        }
        
        return self.report_data['summary']
    
    def print_final_summary(self):
        """Print a comprehensive final summary to console."""
        summary = self.generate_summary()
        
        logging.info("\n" + "="*80)
        logging.info("🎉 WORKFLOW COMPLETION RECAP")
        logging.info("="*80)
        logging.info(f"🗂️  Goal: {self.report_data['summary'].get('workflow_goal', 'N/A')}")
        logging.info(f"⏱️  Duration: {summary['workflow_duration_seconds']:.2f}s")
        logging.info(f"📊 Steps: {summary['successful_steps']}/{summary['total_steps']} succeeded ({summary['success_rate']*100:.1f}%)")
        
        # Performance metrics
        if summary['performance_metrics']:
            logging.info("\n🎯 MODEL PERFORMANCE SNAPSHOT")
            for metric, value in summary['performance_metrics'].items():
                if isinstance(value, float):
                    if 'accuracy' in metric.lower() or 'r2' in metric.lower():
                        logging.info(f"   {metric}: {value:.4f} ({value*100:.2f}%)")
                    else:
                        logging.info(f"   {metric}: {value:.4f}")
                else:
                    logging.info(f"   {metric}: {value}")
        
        # Feature analysis summary
        if summary['feature_analysis']:
            fa = summary['feature_analysis']
            logging.info(f"\n🧠 FEATURE ANALYSIS RECAP")
            logging.info(f"   Features Removed: {fa.get('features_removed', 0)}")
            logging.info(f"   Features Kept: {fa.get('features_kept', 0)}")
        
        # Recommendations summary
        if summary['recommendations']:
            rec = summary['recommendations']
            logging.info(f"\n📋 PRESCRIPTIVE ACTIONS")
            logging.info(f"   Total Recommendations: {rec.get('total_recommendations', 0)}")
            if 'priority_distribution' in rec:
                logging.info("   Priority Distribution:")
                for priority, count in rec['priority_distribution'].items():
                    logging.info(f"     {priority}: {count}")
        
        # Errors summary
        if self.report_data['errors']:
            logging.info(f"\n❌ ERRORS ENCOUNTERED ({len(self.report_data['errors'])})")
            for error in self.report_data['errors']:
                logging.info(f"   • {error['error_type']}: {error['error_message']}")
        else:
            logging.info("\n✅ No errors recorded.")

        logging.info("\n--- Stage Highlights ---")
        for entry in self.report_data.get('stage_summaries', []):
            logging.info(f"• {entry['stage_name']}: {entry['summary']}")

        if self.report_data.get('hitl_events'):
            logging.info("\n👤 HITL INTERACTIONS")
            for event in self.report_data['hitl_events']:
                meta = event.get("metadata") or {}
                meta_text = ", ".join(f"{k}={v}" for k, v in meta.items())
                logging.info(f"   {event['title']} → {event['decision']} ({meta_text})")

        logging.info("\n" + "="*80 + "\n")
    
    def save_report(self, filename: Optional[str] = None):
        """Save the complete report to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        logging.info(f"📄 Complete report saved to: {filepath}")
        return filepath

def create_reporter(output_dir: str = "logs") -> WorkflowReporter:
    """Factory function to create a workflow reporter."""
    return WorkflowReporter(output_dir)

if __name__ == "__main__":
    # Test the reporting system
    reporter = create_reporter()
    
    # Simulate workflow steps
    reporter.log_step("Data Loading", True, "Dataset loaded successfully", 
                     {"shape": (1000, 10), "missing_values": 0}, 2.5)
    
    reporter.log_step("Preprocessing", True, "Data preprocessed", 
                     {"features_removed": 2, "features_kept": 8}, 1.8)
    
    reporter.log_performance_metrics({
        "r2": 0.85,
        "mse": 0.15,
        "accuracy": 0.92
    })
    
    reporter.log_feature_analysis({
        "summary": "Intelligent analysis completed",
        "recommendations": {
            "features_to_remove": [{"feature": "id", "reason": "identifier"}],
            "features_to_keep": [{"feature": "temp", "score": 0.8}]
        }
    })
    
    reporter.print_final_summary()
    reporter.save_report()
