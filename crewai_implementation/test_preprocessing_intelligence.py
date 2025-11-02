#!/usr/bin/env python3
"""
Test script for the Preprocessing Intelligence Agent
"""

import sys
import os
import json
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.preprocessing_intelligence_agent import PreprocessingIntelligenceAgent

def test_preprocessing_intelligence():
    """Test the Preprocessing Intelligence Agent with manufacturing data."""
    
    print("🧪 Testing Preprocessing Intelligence Agent...")
    print("=" * 60)
    
    # Initialize the agent
    try:
        agent = PreprocessingIntelligenceAgent(llm_model="gemini-2.5-flash", temperature=0.1)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Load the manufacturing dataset
    dataset_path = "./data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    try:
        # Load data
        data = pd.read_csv(dataset_path)
        print(f"📊 Loaded dataset: {data.shape[0]} rows × {data.shape[1]} columns")
        
        # Define target and problem type based on our previous analysis
        target_column = "Maintenance_Priority"
        problem_type = "classification"
        
        print(f"🎯 Target: {target_column}, Problem Type: {problem_type}")
        
        # Run preprocessing
        print("\n🔧 Running preprocessing analysis...")
        results = agent.preprocess_dataset(data, target_column=target_column, problem_type=problem_type)
        
        # Display results
        print("\n📈 Preprocessing Results:")
        print("-" * 40)
        
        # Validation results
        validation = results['validation_results']
        print(f"Data Types: {len(validation['numerical_features'])} numerical, {len(validation['categorical_features'])} categorical")
        print(f"Missing Values: {sum(validation['missing_percentage'].values()):.1f}% total")
        print(f"Duplicate Rows: {validation['duplicate_rows']}")
        print(f"High Cardinality Features: {len(validation['high_cardinality_features'])}")
        print(f"Outlier Columns: {len(validation['outlier_columns'])}")
        
        # Feature analysis
        feature_analysis = results['feature_analysis']
        if feature_analysis['status'] == 'completed':
            print(f"\n🔍 Feature Analysis: {feature_analysis['status']}")
            print(f"Summary: {feature_analysis['summary']}")
        else:
            print(f"\n🔍 Feature Analysis: {feature_analysis['status']}")
        
        # Preprocessing pipeline
        pipeline = results['preprocessing_pipeline']
        print(f"\n⚙️ Preprocessing Pipeline:")
        print(f"  Strategy: {pipeline['strategy'].get('tools', [])}")
        print(f"  Numerical Steps: {pipeline['numerical_steps']}")
        print(f"  Categorical Steps: {pipeline['categorical_steps']}")
        
        # Data transformation
        if results['preprocessed_data'] is not None:
            print(f"\n📊 Data Transformation:")
            print(f"  Original Shape: {results['original_data_shape']}")
            print(f"  Preprocessed Shape: {results['preprocessed_data_shape']}")
            print(f"  Features Added: {results['preprocessed_data_shape'][1] - results['original_data_shape'][1]}")
        else:
            print(f"\n❌ Data transformation failed")
        
        # Recommendations
        recommendations = results['recommendations']
        if recommendations:
            print(f"\n🤖 Intelligent Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n✅ Preprocessing analysis completed successfully!")
        
        # Save results
        output_file = "preprocessing_intelligence_results.json"
        
        # Prepare results for JSON serialization (remove non-serializable objects)
        serializable_results = {
            "original_data_shape": results['original_data_shape'],
            "preprocessed_data_shape": results['preprocessed_data_shape'],
            "validation_results": results['validation_results'],
            "feature_analysis": {
                "status": results['feature_analysis']['status'],
                "summary": results['feature_analysis'].get('summary', ''),
                "reason": results['feature_analysis'].get('reason', ''),
                "error": results['feature_analysis'].get('error', '')
            },
            "preprocessing_pipeline": {
                "strategy": results['preprocessing_pipeline']['strategy'],
                "numerical_features": results['preprocessing_pipeline']['numerical_features'],
                "categorical_features": results['preprocessing_pipeline']['categorical_features'],
                "numerical_steps": results['preprocessing_pipeline']['numerical_steps'],
                "categorical_steps": results['preprocessing_pipeline']['categorical_steps'],
                "data_summary": results['preprocessing_pipeline']['data_summary']
            },
            "recommendations": results['recommendations']
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessing_intelligence()
