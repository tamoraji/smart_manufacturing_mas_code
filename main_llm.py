

import logging
from agents.llm_planner_agent import LLMPlannerAgent
import sys
import argparse
import os
import pandas as pd
from utils.schema_discovery import discover_dataset_schema

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [Main-LLM] - %(message)s')


def main():
    """
    Main entry point for the LLM-powered MAS application.
    """
    parser = argparse.ArgumentParser(description="Run the LLM-powered MAS workflow.")
    parser.add_argument('--planner-llm', type=str, default='gemini', help="Planner LLM backend: 'gemini' (default), 'ollama', or 'mock'")
    parser.add_argument('--planner-model', type=str, default=None, help="Model name for planner LLM (if local)")
    parser.add_argument('--decision-llm', type=str, default=None, help="Decision LLM backend: 'ollama', 'mock', or None (default: use planner LLM)")
    parser.add_argument('--decision-model', type=str, default=None, help="Model name for decision LLM (e.g., 'qwen3:4b')")
    parser.add_argument('--dataset', type=str, default=None, help="Path to CSV dataset. If provided with --auto, runs non-interactive.")
    parser.add_argument('--auto', action='store_true', help="Enable non-interactive mode with auto schema discovery and auto-approve HITL.")
    parser.add_argument('--batch', action='store_true', help="Process all datasets found under the data/ directory (implies --auto).")
    parser.add_argument('--interface', type=str, default='cli', help="HITL interface: 'cli' or 'web'.")
    args = parser.parse_args()

    # Choose HITL interface
    from utils.hitl_interface import get_hitl_interface
    hitl_interface = get_hitl_interface(args.interface)

    # Auto-approve HITL if --auto or --batch
    if args.auto or args.batch:
        os.environ["HITL_AUTO"] = "1"

    def auto_select_from_schema(csv_path: str):
        df = pd.read_csv(csv_path)
        schema = discover_dataset_schema(df)
        # Determine problem type and target from suggested_targets if available
        suggested = schema.get('suggested_targets', []) or []
        if suggested:
            best = sorted(suggested, key=lambda x: x.get('score', 0), reverse=True)[0]
            target = best.get('column')
            problem = best.get('suggested_task', 'classification')
        else:
            # Fallback: choose the last column as target if it looks categorical else regression
            last_col = df.columns[-1]
            target = last_col
            if str(df[last_col].dtype) in ['object', 'category'] or df[last_col].nunique() <= 20:
                problem = 'classification'
            else:
                problem = 'regression'
        # Features: all non-identifier/timestamp and not target
        cols_info = schema.get('columns', {})
        feature_cols = [c for c in df.columns if c != target and cols_info.get(c, {}).get('role') not in ['identifier', 'timestamp']]
        return feature_cols, target, problem

    def run_single_dataset(csv_path: str):
        if args.auto:
            feature_cols, target_col, problem_type = auto_select_from_schema(csv_path)
        else:
            feature_cols = target_col = problem_type = None
            dataset_path, feature_cols, target_col, problem_type = LLMPlannerAgent.interactive_setup(hitl_interface)
            csv_path = dataset_path

        goal = f"Load the selected dataset, preprocess it, analyze it to solve a {problem_type} problem, and generate a prescriptive action plan."

        logging.info("Initializing the LLM-powered MAS application...")

        llm_agent = None
        decision_llm_agent = None
        # Planner LLM
        if args.planner_llm == 'ollama':
            from agents.local_llm_agent import LocalLLMAgent
            planner_model = args.planner_model or 'qwen3:4b'
            llm_agent = LocalLLMAgent(backend='ollama', model_name=planner_model)
            logging.info(f"Using local planner LLM agent: ollama, model={planner_model}")
        elif args.planner_llm == 'mock':
            from agents.local_llm_agent import LocalLLMAgent
            llm_agent = LocalLLMAgent(backend='mock')
            logging.info("Using mock planner LLM agent")
        elif args.planner_llm != 'gemini':
            logging.warning(f"Unknown planner LLM backend '{args.planner_llm}', defaulting to Gemini.")

        # Decision LLM
        if args.decision_llm == 'ollama':
            from agents.local_llm_agent import LocalLLMAgent
            decision_model = args.decision_model or 'qwen3:4b'
            decision_llm_agent = LocalLLMAgent(backend='ollama', model_name=decision_model)
            logging.info(f"Using local decision LLM agent: ollama, model={decision_model}")
        elif args.decision_llm == 'mock':
            from agents.local_llm_agent import LocalLLMAgent
            decision_llm_agent = LocalLLMAgent(backend='mock')
            logging.info("Using mock decision LLM agent")
        elif args.decision_llm is not None:
            logging.warning(f"Unknown decision LLM backend '{args.decision_llm}', using planner LLM for all decisions.")

        try:
            # Initialize and run the LLM Planner Agent
            llm_planner = LLMPlannerAgent(
                dataset_path=csv_path,
                feature_columns=feature_cols,
                target_column=target_col,
                problem_type=problem_type,
                llm_agent=llm_agent,
                decision_llm_agent=decision_llm_agent,
                hitl_interface=hitl_interface
            )
            llm_planner.run_workflow_with_llm(goal)
        except Exception as e:
            logging.error(f"An error occurred during the LLM workflow: {e}", exc_info=True)

    # Run in single or batch mode
    if args.batch:
        # Find datasets under data/
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        all_csvs = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith('.csv'):
                    all_csvs.append(os.path.join(root, f))
        logging.info(f"Batch mode: found {len(all_csvs)} datasets")
        for csv in all_csvs:
            logging.info(f"--- Processing dataset: {csv} ---")
            run_single_dataset(csv)
    else:
        if args.dataset is None and args.auto:
            logging.error("--auto requires --dataset. Or use --batch to process all datasets.")
            return
        csv_path = args.dataset
        if not csv_path:
            # Fall back to interactive dataset selection
            dataset_path, feature_cols, target_col, problem_type = LLMPlannerAgent.interactive_setup(hitl_interface)
            csv_path = dataset_path
            # Re-run using the gathered values
            goal = f"Load the selected dataset, preprocess it, analyze it to solve a {problem_type} problem, and generate a prescriptive action plan."
            logging.info("Initializing the LLM-powered MAS application...")
            llm_agent = None
            decision_llm_agent = None
            if args.planner_llm == 'ollama':
                from agents.local_llm_agent import LocalLLMAgent
                planner_model = args.planner_model or 'qwen3:4b'
                llm_agent = LocalLLMAgent(backend='ollama', model_name=planner_model)
                logging.info(f"Using local planner LLM agent: ollama, model={planner_model}")
            elif args.planner_llm == 'mock':
                from agents.local_llm_agent import LocalLLMAgent
                llm_agent = LocalLLMAgent(backend='mock')
                logging.info("Using mock planner LLM agent")
            elif args.planner_llm != 'gemini':
                logging.warning(f"Unknown planner LLM backend '{args.planner_llm}', defaulting to Gemini.")
            if args.decision_llm == 'ollama':
                from agents.local_llm_agent import LocalLLMAgent
                decision_model = args.decision_model or 'qwen3:4b'
                decision_llm_agent = LocalLLMAgent(backend='ollama', model_name=decision_model)
                logging.info(f"Using local decision LLM agent: ollama, model={decision_model}")
            elif args.decision_llm == 'mock':
                from agents.local_llm_agent import LocalLLMAgent
                decision_llm_agent = LocalLLMAgent(backend='mock')
                logging.info("Using mock decision LLM agent")
            elif args.decision_llm is not None:
                logging.warning(f"Unknown decision LLM backend '{args.decision_llm}', using planner LLM for all decisions.")
            try:
                llm_planner = LLMPlannerAgent(
                    dataset_path=csv_path,
                    feature_columns=feature_cols,
                    target_column=target_col,
                    problem_type=problem_type,
                    llm_agent=llm_agent,
                    decision_llm_agent=decision_llm_agent,
                    hitl_interface=hitl_interface
                )
                llm_planner.run_workflow_with_llm(goal)
            except Exception as e:
                logging.error(f"An error occurred during the LLM workflow: {e}", exc_info=True)
        else:
            run_single_dataset(csv_path)
    logging.info("LLM-powered MAS application has finished its run.")

if __name__ == "__main__":
    main()
