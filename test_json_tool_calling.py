#!/usr/bin/env python3
"""
Test script for improved JSON tool-calling in LLMPlannerAgent.
"""

import sys
import os
import pandas as pd
import tempfile
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_planner_agent import LLMPlannerAgent
from agents.local_llm_agent import LocalLLMAgent
from utils.hitl_interface import get_hitl_interface

def test_json_parsing():
    """Test the JSON parsing functionality."""
    print("=== Testing JSON Parsing ===")
    
    planner = LLMPlannerAgent(
        dataset_path="data/test.csv",  # Dummy path
        feature_columns=["feature1"],
        target_column="target",
        problem_type="classification"
    )
    
    # Test valid JSON responses
    test_cases = [
        ('{"tool": "load_and_inspect_data", "reason": "Start with data loading", "finish": false}', True),
        ('{"tool": "analyze_data", "reason": "Time to analyze", "finish": true}', True),
        ('{"tool": "invalid_tool", "reason": "Test", "finish": false}', False),  # Invalid tool
        ('{"tool": "load_and_inspect_data", "reason": "Test"}', False),  # Missing finish
        ('Invalid JSON response', False),  # Invalid JSON
        ('Here is my response: {"tool": "preprocess_data", "reason": "Need preprocessing", "finish": false}', True),  # JSON in text
    ]
    
    success_count = 0
    for i, (test_input, expected_valid) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_input[:50]}...")
        
        # Test JSON parsing
        parsed = planner._parse_json_response(test_input)
        print(f"  Parsed: {parsed}")
        
        # Test validation
        is_valid = parsed and planner._validate_decision(parsed)
        print(f"  Valid: {is_valid} (expected: {expected_valid})")
        
        # For test case 3 (invalid tool), we need to set up the tools context
        if i == 2 and parsed:  # Test case 3 with invalid tool
            planner.tools = ["load_and_inspect_data", "preprocess_data"]  # Set up tools for validation
            is_valid = planner._validate_decision(parsed)
            print(f"  Valid with tools context: {is_valid}")
        
        # Handle special case for invalid JSON (test case 5)
        if i == 4:  # Test case 5: Invalid JSON response
            if parsed is None and not expected_valid:
                success_count += 1
                print("  ✅ PASS")
            else:
                print("  ❌ FAIL")
        else:
            if is_valid == expected_valid:
                success_count += 1
                print("  ✅ PASS")
            else:
                print("  ❌ FAIL")
    
    print(f"\nJSON parsing tests: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def test_mock_llm_workflow():
    """Test workflow with mock LLM that returns structured responses."""
    print("\n=== Testing Mock LLM Workflow ===")
    
    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        test_data.to_csv(f.name, index=False)
        temp_csv_path = f.name
    
    try:
        # Create mock LLM that returns structured responses
        mock_llm = LocalLLMAgent(backend='mock')
        hitl_interface = get_hitl_interface("cli")
        
        planner = LLMPlannerAgent(
            dataset_path=temp_csv_path,
            feature_columns=["feature1", "feature2"],
            target_column="target",
            problem_type="classification",
            llm_agent=mock_llm,
            hitl_interface=hitl_interface
        )
        
        print("Starting workflow with mock LLM...")
        
        # This should work with the mock LLM's structured responses
        try:
            planner.run_workflow_with_llm("Test the workflow with structured tool calling")
            print("✅ Mock LLM workflow completed successfully")
            return True
        except Exception as e:
            print(f"❌ Mock LLM workflow failed: {e}")
            return False
            
    finally:
        # Clean up
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)

def test_prompt_building():
    """Test the workflow prompt building functionality."""
    print("\n=== Testing Prompt Building ===")
    
    planner = LLMPlannerAgent(
        dataset_path="data/test.csv",
        feature_columns=["feature1"],
        target_column="target", 
        problem_type="classification"
    )
    
    # Test context building
    context = {
        "goal": "Test goal",
        "available_tools": ["load_and_inspect_data", "preprocess_data"],
        "completed_steps": [
            {"tool": "load_and_inspect_data", "success": True, "message": "Data loaded", "reason": "Start"}
        ],
        "current_step": 1,
        "last_error": None
    }
    
    prompt = planner._build_workflow_prompt(context)
    print("Generated prompt:")
    print(prompt)
    print("\n✅ Prompt building test completed")
    
    # Check that prompt contains expected elements
    expected_elements = [
        "Goal: Test goal",
        "Available tools:",
        "Completed steps:",
        "You must respond with a valid JSON object",
        "Rules:"
    ]
    
    all_present = all(element in prompt for element in expected_elements)
    print(f"✅ All expected elements present: {all_present}")
    
    return all_present

if __name__ == "__main__":
    print("Testing Improved JSON Tool-Calling...")
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_json_parsing():
        success_count += 1
        
    if test_prompt_building():
        success_count += 1
        
    if test_mock_llm_workflow():
        success_count += 1
    
    print(f"\n=== Test Results ===")
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 All JSON tool-calling tests passed!")
    else:
        print("❌ Some tests failed. Check the logs above.")
