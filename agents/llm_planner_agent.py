import logging
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional, Any, Dict, Tuple
from agents.data_loader_agent import DataLoaderAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent
from agents.optimization_agent import OptimizationAgent
from utils.hitl_interface import get_hitl_interface, HitlInterface
from utils.reporting import create_reporter, WorkflowReporter
from utils.intelligent_summarization import create_summarizer, IntelligentSummarizer
import pandas as pd
import numpy as np
import json

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class LLMPlannerAgent:
    @staticmethod
    def _list_available_datasets():
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        datasets = []
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        datasets.append(os.path.join(folder_path, file))
        return datasets

    @classmethod
    def interactive_setup(cls, hitl_interface: Optional[HitlInterface] = None):
        """
        Interactive setup for dataset, features, target, and problem type.
        Args:
            hitl_interface: HITL interface to use (defaults to CLI)
        Returns:
            tuple: (dataset_path, feature_cols, target_col, problem_type)
        """
        import pandas as pd
        
        if hitl_interface is None:
            hitl_interface = get_hitl_interface("cli")
        
        # List and select dataset
        datasets = cls._list_available_datasets()
        dataset_path = hitl_interface.prompt_with_audit(
            "Select a dataset:",
            options=datasets,
            context={"step": "dataset_selection", "available_datasets": len(datasets)}
        )

        # Read sample data and show columns
        df = pd.read_csv(dataset_path, nrows=100)  # Read sample for column selection
        columns = list(df.columns)
        
        hitl_interface.show_info_with_audit(
            f"Columns in dataset ({len(columns)} total): {columns}",
            context={"step": "column_display", "dataset_path": dataset_path}
        )
        
        # Select problem type first
        problem_types = ['classification', 'regression', 'anomaly_detection']
        problem_type = hitl_interface.prompt_with_audit(
            "Select the problem type:\n"
            "- classification: Predict a categorical outcome (e.g., maintenance priority, efficiency status)\n"
            "- regression: Predict a continuous value\n"
            "- anomaly_detection: Detect unusual patterns in data (no target needed)",
            options=problem_types,
            context={"step": "problem_type_selection", "available_types": problem_types}
        )

        # For supervised learning (classification/regression), select target first
        target_col = None
        if problem_type != 'anomaly_detection':
            target_col = hitl_interface.prompt_with_audit(
                "Select the target column (y):",
                options=columns,
                context={"step": "target_selection", "problem_type": problem_type}
            )
            # Remove target from available feature columns
            columns = [col for col in columns if col != target_col]
            
        # Select features
        feature_prompt = "Select feature columns (Xs):"
        if problem_type == 'anomaly_detection':
            feature_prompt = "Select columns to use for anomaly detection:"
        
        feature_cols = hitl_interface.prompt_with_audit(
            feature_prompt,
            options=columns,
            multi_select=True,
            context={"step": "feature_selection", "problem_type": problem_type, "target_col": target_col}
        )

        return dataset_path, feature_cols, target_col, problem_type
    """
    The LLMPlannerAgent uses a Large Language Model (Google Gemini) to orchestrate the MAS workflow.
    It dynamically decides which agent (tool) to use based on a high-level goal.
    """
    def __init__(self, dataset_path: str, feature_columns: list, target_column: str, problem_type: str,
                 llm_agent: Optional[Any] = None, decision_llm_agent: Optional[Any] = None,
                 hitl_interface: Optional[HitlInterface] = None):
        """
        llm_agent: planner LLM (Gemini or similar) for workflow orchestration and HITL
        decision_llm_agent: tactical LLM (Qwen/Ollama) for preprocessing/model/hyperparameter selection
        hitl_interface: HITL interface for user interactions (defaults to CLI)
        If llm_agent is None, defaults to Gemini. If decision_llm_agent is None, uses planner LLM for all decisions.
        """
        logging.info("Initializing LLM Planner Agent...")
        load_dotenv()
        self.llm_agent = llm_agent
        self.decision_llm_agent = decision_llm_agent
        self.hitl_interface = hitl_interface or get_hitl_interface("cli")
        if self.llm_agent is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("No LLM agent provided and GEMINI_API_KEY not found. Provide a local llm_agent or set GEMINI_API_KEY.")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            logging.info("Using provided planner LLM agent for planning")

        # State management
        self.dataset_path = dataset_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.problem_type = problem_type
        self.full_dataset = None  # Store full dataset before filtering
        self.raw_data = None
        self.preprocessed_data = None
        self.analysis_results = None
        self.recommendations = None
        self.failed_tools = {}  # Track failed tools and their failure counts
        self.reporter = create_reporter()
        self.summarizer = create_summarizer(self.decision_llm_agent)
        self.summarizer.set_logging_mode("minimal")  # Keep all existing logs, add minimal summarizer logs  # Enhanced reporting system

        # Define the tools the LLM can use
        self.tools = {
            "load_and_inspect_data": self._execute_perception_step,
            "preprocess_data": self._execute_preprocessing_step,
            "analyze_data": self._execute_analysis_step,
            "generate_recommendations": self._execute_optimization_step,
        }

        # Internal store for stage summaries emitted after tool execution
        self._stage_snapshot: Dict[str, Dict[str, Any]] = {}

    def run_workflow_with_llm(self, goal: str):
        """
        Executes a dynamic workflow orchestrated by an LLM based on a high-level goal.
        Uses strict JSON parsing with retry logic for robust tool calling.
        """
        logging.info(f"--- Starting LLM-driven Workflow with Goal: '{goal}' ---")
        
        # Log workflow start
        self.reporter.log_step("Workflow Initialization", True, f"Starting workflow with goal: {goal}")
        self.reporter.report_data['workflow_goal'] = goal
        
        # Initialize intelligent summarizer
        self.summarizer.store_workflow_start(
            dataset_path=self.dataset_path,
            problem_type=self.problem_type,
            target_column=self.target_column,
            feature_columns=self.feature_columns
        )
        
        available_tools = list(self.tools.keys())
        max_attempts = 3  # Maximum retry attempts for JSON parsing
        max_steps = len(available_tools) + 2  # Reasonable limit to avoid infinite loops
        
        # Build initial context for the LLM with enhanced reasoning capabilities
        workflow_context = {
            "goal": goal,
            "available_tools": available_tools,
            "completed_steps": [],
            "current_step": 0,
            "performance_insights": {},
            "strategy_adaptations": [],
            "learning_context": {
                "previous_approaches": [],
                "successful_patterns": [],
                "failed_patterns": []
            }
        }
        
        for step in range(max_steps):
            logging.info(f"--- LLM Turn {step+1}/{max_steps} ---")
            
            # Build prompt with context
            prompt = self._build_workflow_prompt(workflow_context)
            logging.info(f"Prompt sent to LLM:\n{prompt}\n---")

            # Get LLM response with retry logic
            decision = self._get_llm_decision(prompt, max_attempts)
            
            if decision is None:
                logging.error("[LLM Planner] Failed to get valid decision from LLM after retries. Halting workflow.")
                break
                
            tool_name = decision.get('tool')
            finish_flag = decision.get('finish', False)
            reason = decision.get('reason', '')
            
            logging.info(f"[LLM Planner] LLM decided: tool='{tool_name}', finish={finish_flag}, reason='{reason}'")
            
            if tool_name not in self.tools:
                logging.warning(f"[LLM Planner] LLM chose invalid tool: '{tool_name}'. Available: {available_tools}")
                # Add error context and retry
                workflow_context["last_error"] = f"Invalid tool '{tool_name}' chosen. Must be one of: {available_tools}"
                continue
                
            # Execute the chosen tool
            logging.info(f"[LLM Planner] Executing tool: '{tool_name}'")
            start_time = time.time()
            success, result_message = self.tools[tool_name]()
            duration = time.time() - start_time
            
            # Enhanced logging with reporter
            self.reporter.log_step(
                f"Tool Execution: {tool_name}", 
                success, 
                result_message,
                {"tool": tool_name, "reason": reason},
                duration
            )
            if success:
                self._emit_stage_summary(tool_name)
            
            logging.info(f"[LLM Planner] Tool '{tool_name}' result: success={success}, message='{result_message}'")
            
            # Update workflow context with enhanced reasoning
            step_data = {
                "tool": tool_name,
                "success": success,
                "message": result_message,
                "reason": reason,
                "duration": duration,
                "step_number": step + 1
            }
            workflow_context["completed_steps"].append(step_data)
            workflow_context["current_step"] = step + 1
            
            # Update learning context
            if success:
                workflow_context["learning_context"]["successful_patterns"].append({
                    "tool": tool_name,
                    "context": f"Step {step + 1}",
                    "outcome": "success"
                })
                # Remove any previous errors on success
                workflow_context.pop("last_error", None)
            else:
                logging.error(f"[LLM Planner] Tool '{tool_name}' failed. Adding to context for next decision.")
                
                # Track failed tools and update learning context
                self.failed_tools[tool_name] = self.failed_tools.get(tool_name, 0) + 1
                workflow_context["learning_context"]["failed_patterns"].append({
                    "tool": tool_name,
                    "context": f"Step {step + 1}",
                    "outcome": "failure",
                    "error": result_message
                })
                
                # Enhanced error context with intelligent recovery suggestions
                error_context = f"Tool '{tool_name}' failed: {result_message}"
                if self.failed_tools[tool_name] >= 2:
                    error_context += f" (Failed {self.failed_tools[tool_name]} times)"
                    # Add intelligent recovery suggestions based on tool type and error
                    if tool_name == "analyze_data" and "R²" in result_message and "negative" in result_message.lower():
                        error_context += " - Consider: 1) Different model types, 2) Feature engineering, 3) Data preprocessing adjustments"
                    elif tool_name == "preprocess_data":
                        error_context += " - Consider: 1) Different encoding strategies, 2) Outlier handling, 3) Feature selection"
                    elif tool_name == "generate_recommendations":
                        error_context += " - Consider: 1) Check analysis results quality, 2) Adjust recommendation thresholds, 3) Review data quality"
                
                workflow_context["last_error"] = error_context
                
            # Check if workflow should complete after executing the tool
            if finish_flag:
                logging.info("[LLM Planner] LLM indicated the workflow is complete after executing tool.")
                break
                
        logging.info("--- LLM-driven Workflow Finished ---")
        
        # Mark workflow completion in summarizer
        self.summarizer.store_workflow_end()
        
        # Generate intelligent summary
        logging.info("\n" + "="*80)
        logging.info("🧠 GENERATING INTELLIGENT SUMMARY...")
        logging.info("="*80)
        
        intelligent_summary = self.summarizer.generate_intelligent_summary()
        logging.info(intelligent_summary)
        
        # Save detailed results
        detailed_results_path = self.summarizer.save_detailed_results()
        logging.info(f"📄 Detailed results saved to: {detailed_results_path}")
        
        # Generate and print comprehensive summary
        self.reporter.print_final_summary()
        
        # Save detailed report
        report_path = self.reporter.save_report()
        publication_paths = self.reporter.save_publication_snapshot()
        
        logging.info("LLM-powered MAS application has finished its run.")
        logging.info(f"📁 Publication snapshot: {publication_paths['json']}")
        if publication_paths.get('csv'):
            logging.info(f"📁 Publication recommendations CSV: {publication_paths['csv']}")

    def _build_workflow_prompt(self, context: Dict[str, Any]) -> str:
        """Build a structured prompt for the LLM with current workflow context."""
        prompt_parts = [
            f"Goal: {context['goal']}",
            f"Available tools: {context['available_tools']}",
            f"Current step: {context['current_step'] + 1}",
        ]
        
        if context.get('completed_steps'):
            prompt_parts.append("Completed steps:")
            for i, step in enumerate(context['completed_steps']):
                status = "✅" if step['success'] else "❌"
                duration_str = f" ({step.get('duration', 0):.2f}s)" if 'duration' in step else ""
                prompt_parts.append(f"  {i+1}. {status} {step['tool']}: {step['message']}{duration_str}")
        
        # Add learning context and performance insights
        if context.get('learning_context'):
            learning = context['learning_context']
            if learning.get('successful_patterns'):
                prompt_parts.append(f"\n📈 Successful patterns: {len(learning['successful_patterns'])} successful operations")
            if learning.get('failed_patterns'):
                prompt_parts.append(f"📉 Failed patterns: {len(learning['failed_patterns'])} failed operations")
        
        if context.get('last_error'):
            prompt_parts.append(f"\n⚠️ Last error: {context['last_error']}")
        
        # Add information about failed tools with enhanced context
        if self.failed_tools:
            failed_info = []
            for tool, count in self.failed_tools.items():
                failed_info.append(f"{tool} (failed {count} times)")
            prompt_parts.append(f"\n🚫 Failed tools: {', '.join(failed_info)}")
            prompt_parts.append("⚠️ WARNING: Avoid repeatedly trying failed tools!")
            
        prompt_parts.extend([
            "",
            "🧠 REASONING PROCESS:",
            "Before choosing a tool, think through the following steps:",
            "1. ANALYZE: What is the current state of the workflow?",
            "2. EVALUATE: What has been accomplished and what remains?",
            "3. IDENTIFY: What are the potential next steps and their purposes?",
            "4. DECIDE: Which tool best serves the current need?",
            "5. JUSTIFY: Why is this the optimal choice at this moment?",
            "",
            "You must respond with a valid JSON object containing:",
            '{"tool": "<tool_name>", "reason": "<detailed explanation with reasoning>", "finish": <true/false>}',
            "",
            "Rules:",
            "- Choose exactly one tool from the available tools list",
            "- Set finish=true only when the goal is fully achieved",
            "- Provide a detailed reason explaining your reasoning process",
            "- If there was an error, suggest a different approach with clear reasoning",
            "- IMPORTANT: If you see 'R²:', 'Accuracy:', or 'Anomalies detected:' in completed steps, the analysis is complete",
            "- Do NOT repeat the same tool if it was already successful in previous steps",
            "- If a tool failed multiple times, try a different tool or set finish=true to end gracefully",
            "- Consider the quality of results: poor performance (R² < 0.1) may require different approaches",
            "- ADAPTIVE INTELLIGENCE: The system will automatically try multiple models if performance is poor",
            "- If you see 'ADAPTIVE INTELLIGENCE' in logs, the system is trying different models for better performance"
        ])
        
        return "\n".join(prompt_parts)

    def _get_llm_decision(self, prompt: str, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        """Get a structured decision from the LLM with retry logic."""
        for attempt in range(max_attempts):
            try:
                if self.llm_agent is not None:
                    # Use local LLM agent
                    response = self.llm_agent.generate(prompt, max_tokens=512)
                    raw_text = response.get('raw', '')
                    logging.info(f"Local LLM response (attempt {attempt+1}): {raw_text}")
                else:
                    # Use Gemini
                    response = self.model.generate_content(prompt)
                    raw_text = response.text.strip()
                    logging.info(f"Gemini response (attempt {attempt+1}): {raw_text}")
                
                # Try to parse JSON from response
                decision = self._parse_json_response(raw_text)
                if decision and self._validate_decision(decision):
                    return decision
                else:
                    logging.warning(f"Invalid decision format on attempt {attempt+1}: {decision}")
                    
            except Exception as e:
                logging.warning(f"Error getting LLM decision on attempt {attempt+1}: {e}")
                
        return None

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response text."""
        import json
        import re
        
        # Try to find JSON object in the response
        # Look for { ... } pattern
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try parsing the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        return None

    def _emit_stage_summary(self, tool_name: str):
        """
        Emit a structured stage summary for the most recent tool run, if available.
        """
        snapshot = self._stage_snapshot.pop(tool_name, None)
        if snapshot:
            self.reporter.log_stage_summary(
                snapshot.get("title", tool_name),
                snapshot.get("summary", ""),
                snapshot.get("stats", {})
            )

    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate that the LLM decision has required fields and valid tool."""
        if not isinstance(decision, dict):
            return False
            
        required_fields = ['tool', 'reason', 'finish']
        if not all(field in decision for field in required_fields):
            return False
            
        if not isinstance(decision['finish'], bool):
            return False
        
        # If finishing, tool can be null
        if decision['finish']:
            return True
            
        if not isinstance(decision['tool'], str) or not decision['tool'].strip():
            return False
            
        # Check if tool is in available tools (if context is available)
        if hasattr(self, 'tools') and decision['tool'] not in self.tools:
            return False
            
        return True

    # Tool implementations (wrapped agent calls)
    def _execute_perception_step(self):
        agent = DataLoaderAgent(self.dataset_path)
        self.raw_data = agent.load_data()
        if self.raw_data is None:
            return False, "Data loading failed."
        
        # Store full dataset before filtering (needed for anomaly detection Machine_ID mapping)
        self.full_dataset = self.raw_data.copy()
            
        # Keep only selected columns (features + target for supervised, just features for anomaly detection)
        if self.problem_type == 'anomaly_detection':
            # For anomaly detection, auto-include ID columns if they exist and weren't selected
            id_columns = [col for col in self.raw_data.columns if 'ID' in col.upper() or col.lower().endswith('_id') or col.lower() == 'id']
            id_columns_to_add = [col for col in id_columns if col not in self.feature_columns]
            if id_columns_to_add:
                logging.info(f"Auto-including ID columns for anomaly detection: {id_columns_to_add}")
                self.raw_data = self.raw_data[self.feature_columns + id_columns_to_add]
            else:
                self.raw_data = self.raw_data[self.feature_columns]
        else:
            self.raw_data = self.raw_data[self.feature_columns + [self.target_column]]
            
        agent.inspect_data()

        feature_list = ", ".join(self.feature_columns[:5])
        if len(self.feature_columns) > 5:
            feature_list += ", …"
        stats = {
            "Rows": f"{self.raw_data.shape[0]:,}",
            "Columns": self.raw_data.shape[1],
            "Selected Features": feature_list if self.feature_columns else "None"
        }
        if self.problem_type != 'anomaly_detection' and self.target_column:
            stats["Target"] = self.target_column
        dataset_name = os.path.basename(self.dataset_path)
        summary = f"Loaded '{dataset_name}' for {self.problem_type} with curated feature set."
        self._stage_snapshot["load_and_inspect_data"] = {
            "title": "Stage · Data Loading & Inspection",
            "summary": summary,
            "stats": stats
        }
        return True, f"Data loaded. Shape: {self.raw_data.shape}"

    def _execute_preprocessing_step(self):
        if self.raw_data is None:
            return False, "Cannot preprocess, raw_data is not loaded."

        # Handle preprocessing based on problem type
        # For anomaly detection, use raw_data directly (may include auto-added ID columns)
        if self.problem_type == 'anomaly_detection':
            analysis_data = self.raw_data  # Includes feature_columns + auto-added ID columns
            target_col = None
            protected_cols = list(self.raw_data.columns)  # Protect all columns including auto-added IDs
        else:
            # For supervised learning, use only feature columns
            data_for_preprocessing = self.raw_data[self.feature_columns]
            target_col = self.target_column
            # Create a temporary dataframe with features + target for analysis only
            analysis_data = self.raw_data[self.feature_columns + [self.target_column]]
            protected_cols = self.feature_columns
        
        # Pass target column and problem type for intelligent feature analysis
        # Pass protected columns so that explicit feature selections are not dropped
        agent = PreprocessingAgent(analysis_data, target_column=target_col, problem_type=self.problem_type,
                                   protected_columns=protected_cols)
        processed_features = agent.preprocess()
        
        if processed_features is None:
            return False, "Preprocessing failed."
        
        # Log intelligent feature analysis results if available
        if hasattr(agent, 'last_feature_insights') and agent.last_feature_insights:
            self.reporter.log_feature_analysis(agent.last_feature_insights)
            # Store in intelligent summarizer
            self.summarizer.store_feature_analysis(agent.last_feature_insights)

        # For supervised learning, include target column
        if self.problem_type != 'anomaly_detection' and self.target_column is not None:
            target = self.raw_data[[self.target_column]]
            self.preprocessed_data = pd.concat([processed_features, target], axis=1)
        else:
            self.preprocessed_data = processed_features

        processed_rows, processed_cols = self.preprocessed_data.shape
        numeric_cols = self.preprocessed_data.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = processed_cols - numeric_cols
        summary = (
            f"Preprocessed data ready ({processed_rows:,} rows × {processed_cols} columns) "
            f"for {self.problem_type} stage."
        )
        stats = {
            "Numeric Columns": numeric_cols,
            "Non-numeric Columns": categorical_cols
        }
        if self.problem_type != 'anomaly_detection' and self.target_column:
            stats["Target Appended"] = "Yes"
        self._stage_snapshot["preprocess_data"] = {
            "title": "Stage · Feature Engineering",
            "summary": summary,
            "stats": stats
        }
            
        return True, f"Preprocessing complete. Shape: {self.preprocessed_data.shape}"

    def _execute_analysis_step(self):
        if self.preprocessed_data is None:
            return False, "Cannot analyze, data is not preprocessed."
        # Before running analysis, if anomaly detection ask LLM for suggested params
        params = None
        if self.problem_type == 'anomaly_detection':
            # Build a short dataset summary to include in the prompt
            summary = self._build_dataset_summary(self.preprocessed_data)
            # Use decision LLM if available, else planner LLM
            params = self._ask_llm_for_anomaly_params(summary, use_decision_llm=True)
            # Ask user to approve or modify suggested params (HITL)
            params = self._human_approve_params(params)

        agent = DynamicAnalysisAgent(self.preprocessed_data, self.target_column, task=self.problem_type, params=params)
        results = agent.run()
        
        # User-in-the-loop decision for poor performance
        adaptive_intelligence_used = False
        tried_models = []
        if results and self._is_poor_performance(results):
            metric_label, metric_value = self._performance_metric(results)
            if metric_label and metric_value is not None:
                self.hitl_interface.show_warning_with_audit(
                    f"Model performance is low ({metric_label} = {metric_value:.3f}).",
                    context={
                        "step": "low_performance_warning",
                        "metric_label": metric_label,
                        "metric_value": float(metric_value),
                        "task": self.problem_type
                    }
                )
                decision = self.hitl_interface.prompt_with_audit(
                    "Would you like to retry with alternative models?",
                    options=["retry", "proceed"],
                    context={
                        "step": "low_performance_decision",
                        "metric_label": metric_label,
                        "metric_value": float(metric_value)
                    }
                ).lower()
                if decision == "retry":
                    logging.info("🧠 User requested adaptive retry due to low performance...")
                    tried_models = agent.tried_models.copy() if hasattr(agent, 'tried_models') else []
                    results = agent.run(force_retry=True)
                    adaptive_intelligence_used = True
                    self.reporter.log_hitl_event(
                        "Model Performance Decision",
                        "retry",
                        {
                            "metric": metric_label,
                            "value": round(float(metric_value), 4) if isinstance(metric_value, (int, float)) else metric_value
                        }
                    )
                    if results is None:
                        return False, "Dynamic analysis failed after retry."
                else:
                    self.hitl_interface.show_info_with_audit(
                        "Proceeding with current model results.",
                        context={
                            "step": "low_performance_proceed",
                            "metric_label": metric_label,
                            "metric_value": float(metric_value)
                        }
                    )
                    self.reporter.log_hitl_event(
                        "Model Performance Decision",
                        "proceed",
                        {
                            "metric": metric_label,
                            "value": round(float(metric_value), 4) if isinstance(metric_value, (int, float)) else metric_value
                        }
                    )
        
        if results is None or ("accuracy" in results and results["accuracy"] is None):
            return False, "Dynamic analysis failed."
        
        # Log performance metrics with enhanced reporting
        performance_metrics = {}
        if 'r2' in results:
            performance_metrics['r2'] = results['r2']
        if 'accuracy' in results:
            performance_metrics['accuracy'] = results['accuracy']
        if 'mse' in results:
            performance_metrics['mse'] = results['mse']
        if 'model_name' in results:
            performance_metrics['model_used'] = results['model_name']
        
        if performance_metrics:
            self.reporter.log_performance_metrics(performance_metrics)
            
        # Store model results in intelligent summarizer
        self.summarizer.store_model_result(
            model_name=results.get('model', 'Unknown'),
            performance=performance_metrics,
            adaptive_intelligence=adaptive_intelligence_used,
            tried_models=tried_models
        )
        # Convert feature importances to DataFrame if available
        feature_importances = None
        if results.get('feature_importances') is not None and results.get('feature_names') is not None:
            import pandas as pd
            feature_importances = pd.DataFrame({
                'feature': results['feature_names'],
                'importance': results['feature_importances']
            }).sort_values(by='importance', ascending=False)
        
        # Store results based on problem type
        if self.problem_type == 'anomaly_detection':
            # Anomaly detection returns different structure
            self.analysis_results = {
                'evaluation': results
            }
        else:
            # Supervised learning results
            self.analysis_results = {
                'evaluation': results,
                'feature_importances': feature_importances,
                'test_data_features': results.get('X_test'),
                'test_predictions': results.get('predictions'),
                'train_predictions': results.get('train_predictions')
            }
        # Show appropriate metrics based on task type
        if self.problem_type == 'regression':
            r2 = results.get('r2', 'N/A')
            mse = results.get('mse', 'N/A')
            msg = f"Dynamic analysis complete. Model: {results.get('model')}, R²: {r2}, MSE: {mse}"
        elif self.problem_type == 'classification':
            accuracy = results.get('accuracy', 'N/A')
            msg = f"Dynamic analysis complete. Model: {results.get('model')}, Accuracy: {accuracy}"
        else:  # anomaly_detection
            n_anomalies = results.get('n_anomalies', 'N/A')
            msg = f"Dynamic analysis complete. Model: {results.get('model')}, Anomalies detected: {n_anomalies}"

        model_name = results.get('model') or results.get('model_name') or getattr(agent, 'model_name', 'Unknown')
        analysis_stats: Dict[str, Any] = {
            "Model": model_name,
            "Samples Evaluated": f"{self.preprocessed_data.shape[0]:,}",
            "Adaptive Retry": "Yes" if adaptive_intelligence_used else "No"
        }
        if self.problem_type == 'classification':
            accuracy_val = results.get('accuracy')
            if isinstance(accuracy_val, (int, float)):
                analysis_stats["Accuracy"] = f"{accuracy_val:.4f}"
        elif self.problem_type == 'regression':
            r2_val = results.get('r2')
            mse_val = results.get('mse')
            if isinstance(r2_val, (int, float)):
                analysis_stats["R²"] = f"{r2_val:.4f}"
            if isinstance(mse_val, (int, float)):
                analysis_stats["MSE"] = f"{mse_val:.6f}"
        else:
            anomalies = results.get('n_anomalies')
            if isinstance(anomalies, (int, float)):
                analysis_stats["Anomalies Detected"] = f"{int(anomalies):,}"
            contamination = None
            if getattr(agent, "model", None) is not None and hasattr(agent.model, "get_params"):
                contamination = agent.model.get_params().get("contamination")
            if contamination is None and isinstance(params, dict):
                contamination = params.get('contamination')
            if isinstance(contamination, (int, float)):
                analysis_stats["Contamination"] = f"{contamination:.4f}"

        summary_text = f"Completed model analysis for {self.problem_type} task."
        self._stage_snapshot["analyze_data"] = {
            "title": "Stage · Modeling & Evaluation",
            "summary": summary_text,
            "stats": analysis_stats
        }
        return True, msg

    def _execute_optimization_step(self):
        if self.analysis_results is None:
            return False, "Cannot optimize, analysis not done."
        
        # Handle different result structures for supervised vs anomaly detection
        if self.problem_type == 'anomaly_detection':
            # For anomaly detection, use results_df directly
            payload = {
                'results_df': self.analysis_results['evaluation'].get('results_df'),
                'anomaly_labels': self.analysis_results['evaluation'].get('anomaly_labels')
            }
        else:
            # For supervised learning, use test_data and predictions
            context = self.raw_data.loc[self.analysis_results['test_data_features'].index]
            payload = {
                'test_data': context,
                'test_predictions': self.analysis_results['test_predictions'],
                'train_predictions': self.analysis_results.get('train_predictions'),
                'feature_importances': self.analysis_results['feature_importances']
            }
        
        agent = OptimizationAgent(payload)
        recommendations = agent.generate_recommendations()
        if recommendations is None:
            return False, "Optimization failed."

        # Generate enhanced summary report
        summary_report = agent.generate_summary_report(recommendations)
        
        # Log recommendations with enhanced reporting
        self.reporter.log_recommendations({
            'recommendations': recommendations,
            'summary_report': summary_report
        })
        
        # Store recommendations in intelligent summarizer
        self.summarizer.store_recommendations({
            'recommendations': recommendations,
            'summary_report': summary_report
        })
        
        # --- Human-in-the-Loop Review Step ---
        self.hitl_interface.show_info_with_audit(
            "===== HUMAN-IN-THE-LOOP REVIEW =====",
            context={"step": "recommendation_review", "num_recommendations": len(recommendations)}
        )
        
        # Show the enhanced summary report
        self.hitl_interface.show_info_with_audit(
            summary_report,
            context={"step": "summary_report"}
        )
        
        if recommendations.empty:
            self.recommendations = recommendations
            return True, "No actions to review."

        # Show detailed recommendations to the user
        display_cols = [col for col in ['Machine_ID', 'Priority_Level', 'Contributing_Factors', 'Recommended_Action', 'Estimated_Cost', 'Timeframe'] if col in recommendations.columns]
        
        self.hitl_interface.show_info_with_audit(
            "Detailed Maintenance Recommendations:",
            data=recommendations[display_cols],
            context={"step": "show_recommendations", "display_cols": display_cols}
        )

        # Ask for human approval
        while True:
            user_input = self.hitl_interface.prompt_with_audit(
                "Approve these recommendations?",
                options=["approve", "modify", "reject"],
                context={"step": "approval_decision", "num_recommendations": len(recommendations)}
            ).lower()
            
            if user_input == 'approve':
                self.hitl_interface.show_info_with_audit(
                    "Recommendations approved.",
                    context={"step": "approval_confirmed"}
                )
                self.reporter.log_hitl_event(
                    "Recommendation Review",
                    "approved",
                    {
                        "actions": len(recommendations),
                        "unique_machines": recommendations['Machine_ID'].nunique() if 'Machine_ID' in recommendations.columns else "N/A"
                    }
                )
                self.recommendations = recommendations
                break
            elif user_input == 'modify':
                self.hitl_interface.show_info_with_audit(
                    "You chose to modify the recommendations.\n"
                    "Please enter the indices (comma-separated) of rows to REMOVE from the plan, or press Enter to skip:",
                    data=recommendations.reset_index()[display_cols + ['index']],
                    context={"step": "modification_request"}
                )
                
                idx_input = self.hitl_interface.prompt_user("Indices to remove (comma-separated, or Enter to skip): ").strip()
                if idx_input:
                    try:
                        idx_list = [int(i) for i in idx_input.split(',') if i.strip().isdigit()]
                        mod_recs = recommendations.drop(recommendations.index[idx_list])
                        self.hitl_interface.show_info_with_audit(
                            "Modified recommendations:",
                            data=mod_recs[display_cols],
                            context={"step": "show_modified_recommendations", "removed_indices": idx_list}
                        )
                        
                        confirm = self.hitl_interface.prompt_with_audit(
                            "Approve modified recommendations?",
                            options=["approve", "reject"],
                            context={"step": "modified_approval", "modified_count": len(mod_recs)}
                        ).lower()
                        
                        if confirm == 'approve':
                            self.hitl_interface.show_info_with_audit(
                                "Modified recommendations approved.",
                                context={"step": "modified_approval_confirmed"}
                            )
                            self.reporter.log_hitl_event(
                                "Recommendation Review",
                                "approved_after_modification",
                                {
                                    "actions": len(mod_recs),
                                    "removed_indices": idx_list
                                }
                            )
                            self.recommendations = mod_recs
                            break
                        else:
                            self.hitl_interface.show_info_with_audit(
                                "Modification not approved. Returning to review.",
                                context={"step": "modification_rejected"}
                            )
                            self.reporter.log_hitl_event(
                                "Recommendation Review",
                                "modification_rejected",
                                {
                                    "removed_indices": idx_list
                                }
                            )
                    except Exception as e:
                        self.hitl_interface.show_info_with_audit(
                            f"Error processing indices: {e}. Please try again.",
                            context={"step": "modification_error", "error": str(e)}
                        )
                else:
                    self.hitl_interface.show_info_with_audit(
                        "No modifications made.",
                        context={"step": "no_modifications"}
                    )
                    self.reporter.log_hitl_event(
                        "Recommendation Review",
                        "modify_no_changes",
                        {}
                    )
            elif user_input == 'reject':
                self.hitl_interface.show_info_with_audit(
                    "Recommendations rejected. No actions will be taken.",
                    context={"step": "recommendations_rejected"}
                )
                self.reporter.log_hitl_event(
                    "Recommendation Review",
                    "rejected",
                    {
                        "actions": len(recommendations)
                    }
                )
                self.recommendations = pd.DataFrame()
                break
            else:
                self.hitl_interface.show_info_with_audit(
                    "Invalid input. Please enter 'approve', 'modify', or 'reject'.",
                    context={"step": "invalid_input", "user_input": user_input}
                )
        rec_df = getattr(self, "recommendations", pd.DataFrame())
        summary = (
            "Generated prescriptive maintenance plan with "
            f"{len(rec_df)} recommendation(s)." if not rec_df.empty else
            "No prescriptive actions were required."
        )
        stats = {
            "Total Recommendations": len(rec_df),
        }
        if not rec_df.empty and 'Machine_ID' in rec_df.columns:
            stats["Unique Machines"] = rec_df['Machine_ID'].nunique()
        if not rec_df.empty:
            top_row = rec_df.iloc[0]
            if 'Machine_ID' in top_row and 'Recommended_Action' in top_row:
                stats["Top Action"] = f"{top_row['Machine_ID']}: {top_row['Recommended_Action']}"
        self._stage_snapshot["generate_recommendations"] = {
            "title": "Stage · Prescriptive Optimization",
            "summary": summary,
            "stats": stats
        }
        return True, "Optimization completed with human review."

    def _is_poor_performance(self, results: Dict[str, Any]) -> bool:
        """
        Check if model performance is poor and requires adaptive intelligence.
        """
        if self.problem_type == "regression":
            r2 = results.get("r2", -float('inf'))
            return r2 < 0.1  # Poor R² score
        elif self.problem_type == "classification":
            accuracy = results.get("accuracy", 0)
            return accuracy < 0.6  # Poor accuracy
        else:  # anomaly_detection
            return False  # No clear performance metric for anomaly detection

    def _performance_metric(self, results: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
        """Return the primary performance metric label and value for the current task."""
        if self.problem_type == "regression":
            r2 = results.get("r2")
            if r2 is not None:
                return "R²", r2
        elif self.problem_type == "classification":
            accuracy = results.get("accuracy")
            if accuracy is not None:
                return "Accuracy", accuracy
        return None, None

    # ---- New helper methods for LLM-driven hyperparameter suggestions and HITL ----
    def _build_dataset_summary(self, df: pd.DataFrame) -> str:
        """Create a compact-yet-informative dataset summary for LLM prompts."""
        num_rows, num_cols = df.shape
        miss_pct = df.isnull().mean().mean() * 100
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(include=['object', 'category'])

        summary_lines = [
            f"rows={num_rows}, cols={num_cols}, missing_pct={miss_pct:.2f}%, "
            f"numeric_cols={numeric_df.shape[1]}, categorical_cols={categorical_df.shape[1]}"
        ]

        approx_outlier_frac = None
        feature_stats = []

        if not numeric_df.empty:
            # Descriptive stats for up to the first five numeric columns
            desc = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).transpose()
            for col in desc.index[:5]:
                stats = desc.loc[col]
                feature_stats.append(
                    f"{col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                    f"q1={stats['25%']:.3f}, median={stats['50%']:.3f}, q3={stats['75%']:.3f}, "
                    f"min={stats['min']:.3f}, max={stats['max']:.3f}"
                )

            # Approximate outlier fraction via z-score > 3 heuristic
            stds = numeric_df.std(ddof=0).replace(0, np.nan)
            if not stds.isna().all():
                zscores = numeric_df.sub(numeric_df.mean()).div(stds)
                approx_outlier_frac = float(((zscores.abs() > 3).any(axis=1)).mean())

        if feature_stats:
            summary_lines.append("feature_stats:\n  " + "\n  ".join(feature_stats))

        if approx_outlier_frac is not None and not np.isnan(approx_outlier_frac):
            summary_lines.append(f"approx_outlier_fraction={approx_outlier_frac:.4f}")

        return "\n".join(summary_lines)

    def _ask_llm_for_anomaly_params(self, dataset_summary: str, use_decision_llm: bool = False) -> Dict[str, Any]:
        """
        Ask the decision LLM (if available) to suggest anomaly detection hyperparameters.
        Expected JSON response: {"contamination": 0.05, "n_estimators": 200, "reason": "..."}
        """
        prompt = (
            "You are configuring an IsolationForest for anomaly detection.\n"
            f"Dataset profile:\n{dataset_summary}\n"
            "Please suggest hyperparameters with this guidance:\n"
            "- Provide 'contamination' as a float between 0.001 and 0.2 whenever the summary offers enough signal (e.g., via approx_outlier_fraction). Use 'auto' only if a numeric estimate is unsafe.\n"
            "- Provide 'n_estimators' as an integer (typical range 100-400) balancing accuracy vs. runtime.\n"
            "- Add a concise 'reason' referencing the dataset characteristics.\n"
            "Respond strictly with JSON, e.g. {\"contamination\": 0.04, \"n_estimators\": 256, \"reason\": \"...\"}."
        )
        llm = self.decision_llm_agent if (use_decision_llm and self.decision_llm_agent is not None) else self.llm_agent
        which = 'decision LLM' if (use_decision_llm and self.decision_llm_agent is not None) else 'planner LLM'
        logging.info(f"Querying {which} for anomaly hyperparameters...")
        if llm is not None:
            resp = llm.generate(prompt)
            parsed = resp.get('parsed')
            if not parsed:
                try:
                    parsed = json.loads(resp.get('raw') or '{}')
                except Exception:
                    parsed = None
            if parsed:
                # sanitize
                raw_cont = parsed.get('contamination', 'auto')
                contamination = raw_cont
                try:
                    if isinstance(raw_cont, str) and raw_cont.strip().lower() != 'auto':
                        contamination = float(raw_cont)
                    elif isinstance(raw_cont, (int, float)):
                        contamination = float(raw_cont)
                except (ValueError, TypeError):
                    contamination = 'auto'

                if isinstance(contamination, (int, float)):
                    contamination = float(np.clip(contamination, 0.001, 0.2))

                n_estimators_raw = parsed.get('n_estimators', 200)
                try:
                    n_estimators = int(float(n_estimators_raw))
                except (TypeError, ValueError):
                    n_estimators = 200

                return {
                    'contamination': contamination,
                    'n_estimators': max(50, n_estimators),
                    'reason': parsed.get('reason', '') + f" (by {which})"
                }
        # Fallback defaults
        return {'contamination': 'auto', 'n_estimators': 200, 'reason': f'default (by {which})'}

    def _human_approve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Present suggested params to the user for approval/modification."""
        self.hitl_interface.show_info_with_audit(
            "LLM suggested anomaly detection parameters:",
            data=params,
            context={"step": "show_anomaly_params", "params": params}
        )
        
        while True:
            ans = self.hitl_interface.prompt_with_audit(
                "Approve these parameters?",
                options=["approve", "modify"],
                context={"step": "anomaly_param_approval", "params": params}
            ).lower()
            
            if ans == 'approve':
                self.hitl_interface.show_info_with_audit(
                    "Parameters approved.",
                    context={"step": "anomaly_params_approved"}
                )
                self.reporter.log_hitl_event(
                    "Anomaly Parameter Approval",
                    "approved",
                    {
                        "contamination": params.get("contamination"),
                        "n_estimators": params.get("n_estimators")
                    }
                )
                return params
            elif ans == 'modify':
                cont = self.hitl_interface.prompt_user("Enter contamination (number or 'auto'): ").strip()
                ne = self.hitl_interface.prompt_user("Enter n_estimators (int): ").strip()
                try:
                    contamination = cont if cont == 'auto' else float(cont)
                    n_estimators = int(ne)
                    modified_params = {'contamination': contamination, 'n_estimators': n_estimators, 'reason': 'user_modified'}
                    
                    self.hitl_interface.show_info_with_audit(
                        "Parameters modified successfully.",
                        data=modified_params,
                        context={"step": "anomaly_params_modified", "original": params, "modified": modified_params}
                    )
                    self.reporter.log_hitl_event(
                        "Anomaly Parameter Approval",
                        "modified",
                        {
                            "contamination": modified_params.get("contamination"),
                            "n_estimators": modified_params.get("n_estimators")
                        }
                    )
                    return modified_params
                except Exception as e:
                    self.hitl_interface.show_info_with_audit(
                        f"Invalid input: {e}. Try again.",
                        context={"step": "anomaly_param_error", "error": str(e)}
                    )
            else:
                self.hitl_interface.show_info_with_audit(
                    "Please enter 'approve' or 'modify'.",
                    context={"step": "invalid_anomaly_param_input", "user_input": ans}
                )
