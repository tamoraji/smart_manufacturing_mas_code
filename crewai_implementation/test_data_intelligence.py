#!/usr/bin/env python3
"""
Test script for the Data Intelligence Agent
"""

import sys
import os
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_intelligence_agent import DataIntelligenceAgent

def test_data_intelligence_agent():
    """Test the Data Intelligence Agent with sample data."""
    
    print("🧪 Testing Data Intelligence Agent...")
    print("=" * 50)
    
    # Initialize the agent
    try:
        agent = DataIntelligenceAgent(llm_model="gemini-2.5-flash", temperature=0.1)
        print("✅ Agent initialized successfully with Gemini 2.5 Flash")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test with the smart maintenance dataset
    dataset_path = "./data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Available datasets:")
        data_dir = "../../data"
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.csv'):
                        print(f"  - {os.path.join(root, file)}")
        return
    
    # Test different problem types
    problem_types = ['classification', 'regression', 'anomaly_detection']
    
    for problem_type in problem_types:
        print(f"\n{'='*60}")
        print(f"🎯 Testing {problem_type.upper()} Problem Analysis")
        print(f"{'='*60}")
        
        try:
            # Run the analysis with specific problem type
            results = agent.analyze_dataset(dataset_path, problem_type=problem_type)
            
            if "error" in results:
                print(f"❌ Analysis failed: {results['error']}")
                continue
            
            # Display results
            print(f"\n📈 {problem_type.title()} Analysis Results:")
            print("-" * 40)
            
            # Basic info
            basic_info = results['basic_info']
            print(f"Dataset Shape: {basic_info['num_rows']} rows × {basic_info['num_columns']} columns")
            
            # Problem-aware target suggestions
            if 'problem_aware_targets' in results['schema_discovery']:
                targets = results['schema_discovery']['problem_aware_targets']
                print(f"\n🎯 Problem-Aware Target Suggestions for {problem_type}:")
                for target in targets['suggestions'][:3]:
                    print(f"  - {target['column']} (score: {target['score']})")
                    print(f"    Reasons: {', '.join(target['reasons'][:2])}")
            
            # Feature analysis
            if 'feature_analysis' in results:
                feature_analysis = results['feature_analysis']
                print(f"\n🔧 Feature Analysis for {problem_type}:")
                print(f"  Total Features: {feature_analysis.get('feature_count', 'N/A')}")
                if 'feature_analysis' in feature_analysis:
                    feat_info = feature_analysis['feature_analysis']
                    print(f"  Numeric Features: {feat_info.get('numeric_features', {}).get('count', 0)}")
                    print(f"  Categorical Features: {feat_info.get('categorical_features', {}).get('count', 0)}")
                
                if feature_analysis.get('preprocessing_recommendations'):
                    print(f"\n📋 Preprocessing Recommendations:")
                    for rec in feature_analysis['preprocessing_recommendations'][:3]:
                        print(f"  - {rec}")
            
            # Data quality
            quality = results['data_quality']
            print(f"\n📊 Data Quality:")
            print(f"  Completeness: {quality['completeness']['completeness_rate']:.1f}%")
            print(f"  Duplicates: {quality['uniqueness']['duplicate_percentage']:.1f}%")
            
            # Recommendations
            recommendations = results['recommendations']
            if recommendations:
                print(f"\n💡 {problem_type.title()}-Specific Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"  {i}. {rec}")
            
            print(f"\n✅ {problem_type.title()} analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ {problem_type.title()} analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test auto-detection (no problem type specified)
    print(f"\n{'='*60}")
    print(f"🔍 Testing AUTO-DETECTION Analysis")
    print(f"{'='*60}")
    
    try:
        results = agent.analyze_dataset(dataset_path)
        
        if "error" in results:
            print(f"❌ Auto-detection analysis failed: {results['error']}")
        else:
            print("✅ Auto-detection analysis completed successfully!")
            
            # Show general suggestions
            schema = results['schema_discovery']
            if 'suggested_targets' in schema and schema['suggested_targets']:
                print(f"\n🎯 General Target Suggestions:")
                for target in schema['suggested_targets'][:3]:
                    print(f"  - {target['column']} (score: {target.get('score', 'N/A')})")
        
        # Save results to file
        output_file = "data_intelligence_problem_aware_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Auto-detection analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_intelligence_agent()
