#!/usr/bin/env python3
"""
Test script for integrated ToolDecider functionality.
"""

import pandas as pd
import numpy as np
import logging
from utils.tool_decider import get_tool_decider, create_data_summary
from agents.preprocessing_agent import PreprocessingAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

def test_preprocessing_integration():
    """Test PreprocessingAgent with ToolDecider integration."""
    print("=== Testing PreprocessingAgent with ToolDecider ===")
    
    # Create test data with mixed types and missing values
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5, 6, 7, 8],
        'feature2': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'feature4': [100, 200, 300, 400, 500, 600, 700, 800]
    })
    
    print(f"Original data shape: {test_data.shape}")
    print(f"Missing values: {test_data.isnull().sum().sum()}")
    
    # Test with rule-based decider
    rule_decider = get_tool_decider("rule_based")
    preprocessing_agent = PreprocessingAgent(test_data, tool_decider=rule_decider)
    
    # Test preprocessing
    preprocessed_data = preprocessing_agent.preprocess()
    
    if preprocessed_data is not None:
        print(f"✅ Preprocessing successful! Shape: {preprocessed_data.shape}")
        print(f"✅ No missing values in preprocessed data: {preprocessed_data.isnull().sum().sum()}")
        return True
    else:
        print("❌ Preprocessing failed")
        return False

def test_analysis_integration():
    """Test DynamicAnalysisAgent with ToolDecider integration."""
    print("\n=== Testing DynamicAnalysisAgent with ToolDecider ===")
    
    # Create test data for classification
    np.random.seed(42)
    n_samples = 100
    test_data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.choice([0, 1, 2], n_samples)
    })
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Target distribution: {test_data['target'].value_counts().to_dict()}")
    
    # Test with rule-based decider
    rule_decider = get_tool_decider("rule_based")
    analysis_agent = DynamicAnalysisAgent(
        test_data, 
        target_column='target', 
        task='classification',
        tool_decider=rule_decider
    )
    
    # Test analysis
    results = analysis_agent.run()
    
    if results is not None:
        print(f"✅ Analysis successful! Model: {results.get('model')}")
        print(f"✅ Accuracy: {results.get('accuracy', 'N/A')}")
        return True
    else:
        print("❌ Analysis failed")
        return False

def test_regression_integration():
    """Test DynamicAnalysisAgent with regression task."""
    print("\n=== Testing DynamicAnalysisAgent with Regression ===")
    
    # Create test data for regression
    np.random.seed(42)
    n_samples = 100
    test_data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randn(n_samples) * 2 + 5  # Continuous target
    })
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Target range: [{test_data['target'].min():.2f}, {test_data['target'].max():.2f}]")
    
    # Test with rule-based decider
    rule_decider = get_tool_decider("rule_based")
    analysis_agent = DynamicAnalysisAgent(
        test_data, 
        target_column='target', 
        task='regression',
        tool_decider=rule_decider
    )
    
    # Test analysis
    results = analysis_agent.run()
    
    if results is not None:
        print(f"✅ Regression analysis successful! Model: {results.get('model')}")
        print(f"✅ R² Score: {results.get('r2', 'N/A')}")
        print(f"✅ MSE: {results.get('mse', 'N/A')}")
        return True
    else:
        print("❌ Regression analysis failed")
        return False

def test_anomaly_detection():
    """Test DynamicAnalysisAgent with anomaly detection."""
    print("\n=== Testing DynamicAnalysisAgent with Anomaly Detection ===")
    
    # Create test data with some anomalies
    np.random.seed(42)
    n_samples = 100
    normal_data = np.random.randn(n_samples, 3)
    anomaly_data = np.random.randn(5, 3) * 3 + 5  # Outliers
    
    all_data = np.vstack([normal_data, anomaly_data])
    test_data = pd.DataFrame(all_data, columns=['feature1', 'feature2', 'feature3'])
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test with rule-based decider
    rule_decider = get_tool_decider("rule_based")
    analysis_agent = DynamicAnalysisAgent(
        test_data, 
        target_column=None, 
        task='anomaly_detection',
        tool_decider=rule_decider
    )
    
    # Test analysis
    results = analysis_agent.run()
    
    if results is not None:
        print(f"✅ Anomaly detection successful! Model: {results.get('model')}")
        print(f"✅ Anomalies detected: {results.get('n_anomalies', 'N/A')}")
        return True
    else:
        print("❌ Anomaly detection failed")
        return False

if __name__ == "__main__":
    print("Testing ToolDecider Integration...")
    
    success_count = 0
    total_tests = 4
    
    # Run all tests
    if test_preprocessing_integration():
        success_count += 1
    
    if test_analysis_integration():
        success_count += 1
        
    if test_regression_integration():
        success_count += 1
        
    if test_anomaly_detection():
        success_count += 1
    
    print(f"\n=== Test Results ===")
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 All integration tests passed!")
    else:
        print("❌ Some tests failed. Check the logs above.")
