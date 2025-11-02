import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class IntelligentSummarizer:
    """
    LLM-powered intelligent summarization of model results and workflow execution.
    Stores detailed results and provides clean, professional summaries.
    """
    
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
        self.stored_results = {
            'workflow_start_time': None,
            'workflow_end_time': None,
            'dataset_info': {},
            'preprocessing_steps': [],
            'model_results': [],
            'feature_analysis': {},
            'recommendations': {},
            'performance_metrics': {},
            'adaptive_intelligence_events': [],
            'errors': [],
            'summary': None
        }
        self.logging_enabled = True
        
    def store_workflow_start(self, dataset_path: str, problem_type: str, target_column: str, feature_columns: List[str]):
        """Store workflow initialization information."""
        self.stored_results['workflow_start_time'] = datetime.now().isoformat()
        self.stored_results['dataset_info'] = {
            'dataset_path': dataset_path,
            'problem_type': problem_type,
            'target_column': target_column,
            'feature_columns': feature_columns
        }
        self._log_info("🚀 Workflow started", f"Dataset: {os.path.basename(dataset_path)}")
        
    def store_preprocessing_step(self, step_name: str, details: Dict[str, Any], duration: float = None):
        """Store preprocessing step information."""
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'step_name': step_name,
            'details': details,
            'duration': duration
        }
        self.stored_results['preprocessing_steps'].append(step_data)
        self._log_info(f"🔧 Preprocessing: {step_name}", f"Duration: {duration:.2f}s" if duration else "")
        
    def store_model_result(self, model_name: str, performance: Dict[str, Any], 
                          adaptive_intelligence: bool = False, tried_models: List[str] = None):
        """Store model training and evaluation results."""
        model_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'performance': performance,
            'adaptive_intelligence_used': adaptive_intelligence,
            'tried_models': tried_models or []
        }
        self.stored_results['model_results'].append(model_data)
        
        # Store adaptive intelligence events separately
        if adaptive_intelligence and tried_models:
            self.stored_results['adaptive_intelligence_events'].append({
                'timestamp': datetime.now().isoformat(),
                'trigger_model': model_name,
                'tried_models': tried_models,
                'final_model': model_name,
                'performance_improvement': performance
            })
        
        # Only log essential information
        if self.logging_enabled:
            if adaptive_intelligence:
                self._log_info(f"🧠 Adaptive Intelligence: {model_name}", 
                             f"Performance: {self._format_performance(performance)}")
            else:
                self._log_info(f"🤖 Model: {model_name}", 
                             f"Performance: {self._format_performance(performance)}")
    
    def store_feature_analysis(self, analysis: Dict[str, Any]):
        """Store intelligent feature analysis results."""
        self.stored_results['feature_analysis'] = analysis
        self._log_info("🧠 Feature Analysis", "Completed intelligent feature analysis")
        
    def store_recommendations(self, recommendations: Dict[str, Any]):
        """Store prescriptive recommendations."""
        self.stored_results['recommendations'] = recommendations
        self._log_info("🎯 Recommendations", "Generated prescriptive recommendations")
        
    def store_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Store error information."""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': error_message,
            'context': context or {}
        }
        self.stored_results['errors'].append(error_data)
        self._log_error(f"❌ {error_type}", error_message)
        
    def store_workflow_end(self):
        """Mark workflow completion."""
        self.stored_results['workflow_end_time'] = datetime.now().isoformat()
        self._log_info("✅ Workflow Complete", "All steps completed successfully")
        
    def generate_intelligent_summary(self) -> str:
        """Generate an intelligent summary using LLM."""
        if not self.llm_agent:
            return self._generate_basic_summary()
            
        try:
            summary_prompt = self._build_summary_prompt()
            response = self.llm_agent.generate(summary_prompt, max_tokens=1000)
            
            if response and 'summary' in response:
                self.stored_results['summary'] = response['summary']
                return response['summary']
            else:
                return self._generate_basic_summary()
                
        except Exception as e:
            logging.warning(f"LLM summarization failed: {e}")
            return self._generate_basic_summary()
    
    def _build_summary_prompt(self) -> str:
        """Build a prompt for LLM-based technical summarization."""
        workflow_data = self.stored_results
        
        prompt_parts = [
            "You are an expert data scientist and ML engineer analyzing a machine learning workflow execution.",
            "Provide a detailed TECHNICAL summary focusing on performance, issues, and actionable insights:",
            "",
            "WORKFLOW INFORMATION:",
            f"Dataset: {workflow_data['dataset_info'].get('dataset_path', 'Unknown')}",
            f"Problem Type: {workflow_data['dataset_info'].get('problem_type', 'Unknown')}",
            f"Target Column: {workflow_data['dataset_info'].get('target_column', 'Unknown')}",
            f"Features: {len(workflow_data['dataset_info'].get('feature_columns', []))} columns",
            "",
            "MODEL PERFORMANCE ANALYSIS:"
        ]
        
        # Add detailed model results
        for i, model_result in enumerate(workflow_data['model_results'], 1):
            model_name = model_result['model_name']
            performance = model_result['performance']
            adaptive = model_result['adaptive_intelligence_used']
            tried_models = model_result.get('tried_models', [])
            
            prompt_parts.append(f"Model {i}: {model_name}")
            if 'r2' in performance:
                r2_val = performance['r2']
                if r2_val == float('-inf'):
                    prompt_parts.append(f"  R² Score: FAILED (infinite value - likely data issue)")
                else:
                    prompt_parts.append(f"  R² Score: {r2_val:.4f} ({'EXCELLENT' if r2_val > 0.8 else 'GOOD' if r2_val > 0.5 else 'POOR' if r2_val > 0.1 else 'FAILED'})")
            if 'accuracy' in performance:
                acc_val = performance['accuracy']
                prompt_parts.append(f"  Accuracy: {acc_val:.4f} ({'EXCELLENT' if acc_val > 0.9 else 'GOOD' if acc_val > 0.7 else 'POOR' if acc_val > 0.5 else 'FAILED'})")
            if 'mse' in performance:
                prompt_parts.append(f"  MSE: {performance['mse']:.4f}")
            if adaptive:
                prompt_parts.append(f"  🧠 ADAPTIVE INTELLIGENCE ACTIVATED")
                prompt_parts.append(f"  Tried Models: {', '.join(tried_models) if tried_models else 'None'}")
            prompt_parts.append("")
        
        # Add feature analysis
        if workflow_data['feature_analysis']:
            prompt_parts.extend([
                "FEATURE ANALYSIS:",
                f"Features Analyzed: {len(workflow_data['feature_analysis'].get('recommendations', {}).get('features_to_keep', []))}",
                f"Features Removed: {len(workflow_data['feature_analysis'].get('recommendations', {}).get('features_to_remove', []))}",
                ""
            ])
        
        # Add recommendations
        if workflow_data['recommendations']:
            recs = workflow_data['recommendations'].get('recommendations', [])
            if hasattr(recs, 'empty'):  # DataFrame
                rec_count = len(recs) if not recs.empty else 0
            else:
                rec_count = len(recs) if recs else 0
            prompt_parts.extend([
                "RECOMMENDATIONS:",
                f"Total Recommendations: {rec_count}",
                ""
            ])
        
        # Add errors if any
        if workflow_data['errors']:
            prompt_parts.extend([
                "ERRORS:",
                f"Total Errors: {len(workflow_data['errors'])}",
                ""
            ])
        
        prompt_parts.extend([
            "TECHNICAL ANALYSIS REQUIRED:",
            "Provide a detailed technical summary covering:",
            "",
            "1. WORKFLOW STATUS:",
            "   - Success/failure rate and critical issues",
            "   - Data quality and preprocessing effectiveness",
            "   - Model performance analysis with specific metrics",
            "",
            "2. ADAPTIVE INTELLIGENCE ANALYSIS:",
            "   - Whether adaptive intelligence was triggered and why",
            "   - Models tried and their individual performance",
            "   - Root cause of any failures",
            "",
            "3. FEATURE ENGINEERING INSIGHTS:",
            "   - Features removed/kept and reasoning",
            "   - Data leakage prevention measures",
            "   - Correlation and multicollinearity issues",
            "",
            "4. TECHNICAL ISSUES & DIAGNOSIS:",
            "   - Specific error analysis and root causes",
            "   - Data quality problems identified",
            "   - Model convergence issues",
            "",
            "5. ACTIONABLE RECOMMENDATIONS:",
            "   - Immediate fixes needed",
            "   - Data preprocessing improvements",
            "   - Alternative modeling approaches",
            "",
            "Format as detailed technical report with specific metrics, error codes, and actionable insights.",
            "Respond with JSON: {\"summary\": \"detailed technical summary here\"}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_basic_summary(self) -> str:
        """Generate a basic summary without LLM."""
        workflow_data = self.stored_results
        
        summary_lines = [
            "=" * 60,
            "🎯 INTELLIGENT WORKFLOW SUMMARY",
            "=" * 60,
            f"📊 Dataset: {os.path.basename(workflow_data['dataset_info'].get('dataset_path', 'Unknown'))}",
            f"🎯 Problem Type: {workflow_data['dataset_info'].get('problem_type', 'Unknown')}",
            f"📈 Features: {len(workflow_data['dataset_info'].get('feature_columns', []))} columns",
            ""
        ]
        
        # Model performance
        if workflow_data['model_results']:
            summary_lines.append("🤖 MODEL PERFORMANCE:")
            for i, model_result in enumerate(workflow_data['model_results'], 1):
                model_name = model_result['model_name']
                performance = model_result['performance']
                adaptive = model_result['adaptive_intelligence_used']
                
                summary_lines.append(f"  {i}. {model_name}")
                if 'r2' in performance:
                    summary_lines.append(f"     R² Score: {performance['r2']:.4f} ({performance['r2']*100:.1f}%)")
                if 'accuracy' in performance:
                    summary_lines.append(f"     Accuracy: {performance['accuracy']:.4f} ({performance['accuracy']*100:.1f}%)")
                if 'mse' in performance:
                    summary_lines.append(f"     MSE: {performance['mse']:.4f}")
                if adaptive:
                    summary_lines.append(f"     🧠 Adaptive Intelligence: Tried {len(model_result.get('tried_models', []))} models")
                summary_lines.append("")
        
        # Feature analysis
        if workflow_data['feature_analysis']:
            recs = workflow_data['feature_analysis'].get('recommendations', {})
            summary_lines.extend([
                "🧠 FEATURE ANALYSIS:",
                f"  Features Kept: {len(recs.get('features_to_keep', []))}",
                f"  Features Removed: {len(recs.get('features_to_remove', []))}",
                ""
            ])
        
        # Recommendations
        if workflow_data['recommendations']:
            recs = workflow_data['recommendations'].get('recommendations', [])
            if hasattr(recs, 'empty'):  # DataFrame
                rec_count = len(recs) if not recs.empty else 0
            else:
                rec_count = len(recs) if recs else 0
            summary_lines.extend([
                "🎯 RECOMMENDATIONS:",
                f"  Total Generated: {rec_count}",
                ""
            ])
        
        # Errors
        if workflow_data['errors']:
            summary_lines.extend([
                "⚠️ ISSUES:",
                f"  Errors Encountered: {len(workflow_data['errors'])}",
                ""
            ])
        
        # Duration
        if workflow_data['workflow_start_time'] and workflow_data['workflow_end_time']:
            start_time = datetime.fromisoformat(workflow_data['workflow_start_time'])
            end_time = datetime.fromisoformat(workflow_data['workflow_end_time'])
            duration = (end_time - start_time).total_seconds()
            summary_lines.append(f"⏱️ Total Duration: {duration:.2f} seconds")
        
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
    
    def _format_performance(self, performance: Dict[str, Any]) -> str:
        """Format performance metrics for logging."""
        if 'r2' in performance:
            return f"R²: {performance['r2']:.4f}"
        elif 'accuracy' in performance:
            return f"Accuracy: {performance['accuracy']:.4f}"
        else:
            return "Performance metrics available"
    
    def _log_info(self, title: str, message: str = ""):
        """Log information with clean formatting."""
        if self.logging_enabled:
            if message:
                logging.info(f"[Summarizer] {title}: {message}")
            else:
                logging.info(f"[Summarizer] {title}")
    
    def _log_error(self, title: str, message: str):
        """Log error with clean formatting."""
        if self.logging_enabled:
            logging.error(f"[Summarizer] {title}: {message}")
    
    def save_detailed_results(self, filepath: str = None) -> str:
        """Save detailed results to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/detailed_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.stored_results, f, indent=4, default=str)
        
        return filepath
    
    def disable_logging(self):
        """Disable verbose logging during workflow execution."""
        self.logging_enabled = False
    
    def enable_logging(self):
        """Re-enable logging."""
        self.logging_enabled = True
    
    def set_logging_mode(self, mode: str = "verbose"):
        """
        Set logging mode for the summarizer.
        - "verbose": Log all information (default)
        - "minimal": Only log essential information
        - "silent": No logging from summarizer
        """
        if mode == "verbose":
            self.logging_enabled = True
        elif mode == "minimal":
            self.logging_enabled = True
        elif mode == "silent":
            self.logging_enabled = False
        else:
            self.logging_enabled = True

def create_summarizer(llm_agent=None) -> IntelligentSummarizer:
    """Factory function to create an IntelligentSummarizer instance."""
    return IntelligentSummarizer(llm_agent)
