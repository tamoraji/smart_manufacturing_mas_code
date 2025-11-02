"""
Dataset Schema Discovery for automatic preprocessing suggestions.
This module analyzes datasets to automatically detect column roles, data types,
and suggest appropriate preprocessing strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class ColumnRoleDetector:
    """Detects the role of columns in a dataset (identifier, target, feature, etc.)."""
    
    # Common identifier patterns
    ID_PATTERNS = [
        r'.*id$', r'.*_id$', r'^id.*', r'^ID.*', r'.*key$', r'.*_key$',
        r'.*index$', r'.*_index$', r'^index.*', r'.*code$', r'.*_code$'
    ]
    
    # Common timestamp patterns
    TIME_PATTERNS = [
        r'.*time.*', r'.*date.*', r'.*timestamp.*', r'.*created.*', r'.*updated.*',
        r'.*_at$', r'.*_on$', r'^time.*', r'^date.*'
    ]
    
    # Common target patterns
    TARGET_PATTERNS = [
        r'.*target.*', r'.*label.*', r'.*class.*', r'.*category.*', r'.*outcome.*',
        r'.*result.*', r'.*score.*', r'.*rating.*', r'.*priority.*', r'.*status.*'
    ]
    
    # Common feature patterns
    FEATURE_PATTERNS = [
        r'.*feature.*', r'.*value.*', r'.*measure.*', r'.*metric.*', r'.*count.*',
        r'.*amount.*', r'.*rate.*', r'.*ratio.*', r'.*percentage.*', r'.*level.*'
    ]

    @classmethod
    def detect_column_role(cls, column_name: str, data: pd.Series) -> str:
        """
        Detect the role of a column based on name patterns and data characteristics.
        Returns: 'identifier', 'timestamp', 'target', 'feature', or 'unknown'
        """
        col_lower = column_name.lower()
        
        # Check for identifier patterns
        for pattern in cls.ID_PATTERNS:
            if re.match(pattern, col_lower):
                return 'identifier'
        
        # Check for timestamp patterns
        for pattern in cls.TIME_PATTERNS:
            if re.match(pattern, col_lower):
                return 'timestamp'
        
        # Check for target patterns
        for pattern in cls.TARGET_PATTERNS:
            if re.match(pattern, col_lower):
                return 'target'
        
        # Check data characteristics
        if cls._is_identifier_column(data):
            return 'identifier'
        
        if cls._is_timestamp_column(data):
            return 'timestamp'
        
        # Default to feature
        return 'feature'

    @classmethod
    def _is_identifier_column(cls, data: pd.Series) -> bool:
        """Check if column is likely an identifier."""
        if data.dtype == 'object':
            # Check if all values are unique (common for IDs)
            if data.nunique() == len(data):
                return True
            # Check if values look like IDs (alphanumeric patterns)
            sample_values = data.dropna().head(10)
            if len(sample_values) > 0:
                id_pattern = re.compile(r'^[A-Za-z0-9_-]+$')
                if all(id_pattern.match(str(val)) for val in sample_values):
                    return True
        return False

    @classmethod
    def _is_timestamp_column(cls, data: pd.Series) -> bool:
        """Check if column is likely a timestamp."""
        if pd.api.types.is_datetime64_any_dtype(data):
            return True
        
        if data.dtype == 'object':
            # Try to parse as datetime
            sample_values = data.dropna().head(5)
            if len(sample_values) > 0:
                try:
                    pd.to_datetime(sample_values)
                    return True
                except:
                    pass
        return False

class DataTypeAnalyzer:
    """Analyzes data types and suggests appropriate preprocessing."""
    
    @classmethod
    def analyze_column_type(cls, column_name: str, data: pd.Series) -> Dict[str, Any]:
        """
        Analyze a column and return detailed type information.
        Returns dict with: type, subtype, issues, suggestions
        """
        analysis = {
            'name': column_name,
            'type': 'unknown',
            'subtype': 'unknown',
            'issues': [],
            'suggestions': [],
            'stats': {}
        }
        
        # Basic statistics
        analysis['stats'] = {
            'count': len(data),
            'null_count': data.isnull().sum(),
            'null_percentage': (data.isnull().sum() / len(data)) * 100,
            'unique_count': data.nunique(),
            'unique_percentage': (data.nunique() / len(data)) * 100
        }
        
        # Detect issues
        if analysis['stats']['null_percentage'] > 50:
            analysis['issues'].append('High missing value percentage')
        elif analysis['stats']['null_percentage'] > 10:
            analysis['issues'].append('Moderate missing values')
        
        if analysis['stats']['unique_percentage'] == 100:
            analysis['issues'].append('All values unique - likely identifier')
        elif analysis['stats']['unique_percentage'] < 5:
            analysis['issues'].append('Very low cardinality')
        
        # Analyze by data type
        if pd.api.types.is_numeric_dtype(data):
            numeric_analysis = cls._analyze_numeric_column(data, analysis)
            analysis.update(numeric_analysis)
        elif pd.api.types.is_datetime64_any_dtype(data):
            datetime_analysis = cls._analyze_datetime_column(data, analysis)
            analysis.update(datetime_analysis)
        elif data.dtype == 'object':
            object_analysis = cls._analyze_object_column(data, analysis)
            analysis.update(object_analysis)
        else:
            analysis['type'] = 'other'
            analysis['subtype'] = str(data.dtype)
        
        return analysis

    @classmethod
    def _analyze_numeric_column(cls, data: pd.Series, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze numeric columns."""
        analysis = {'type': 'numeric'}
        
        # Check if it's integer or float
        if pd.api.types.is_integer_dtype(data):
            analysis['subtype'] = 'integer'
        else:
            analysis['subtype'] = 'float'
        
        # Get numeric statistics
        numeric_data = pd.to_numeric(data, errors='coerce')
        if 'stats' not in analysis:
            analysis['stats'] = {}
        analysis['stats'].update({
            'min': numeric_data.min(),
            'max': numeric_data.max(),
            'mean': numeric_data.mean(),
            'std': numeric_data.std(),
            'median': numeric_data.median()
        })
        
        # Initialize issues and suggestions if not present
        if 'issues' not in analysis:
            analysis['issues'] = []
        if 'suggestions' not in analysis:
            analysis['suggestions'] = []
        
        # Check for issues
        if numeric_data.std() == 0:
            analysis['issues'].append('No variance - constant values')
        
        if numeric_data.min() == numeric_data.max():
            analysis['issues'].append('Single value - no variation')
        
        # Check for outliers (values beyond 3 standard deviations)
        outlier_count = 0
        if not numeric_data.std() == 0:
            z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
            outlier_count = (z_scores > 3).sum()
            if outlier_count > 0:
                analysis['issues'].append(f'{outlier_count} potential outliers')
        
        # Generate suggestions
        if 'null_percentage' in analysis.get('stats', {}) and analysis['stats']['null_percentage'] > 0:
            analysis['suggestions'].append('Consider imputation strategy')
        
        if 'std' in analysis.get('stats', {}) and analysis['stats']['std'] > 0 and analysis['stats']['std'] > analysis['stats'].get('mean', 0):
            analysis['suggestions'].append('High variance - consider scaling')
        
        if outlier_count > len(data) * 0.05:  # More than 5% outliers
            analysis['suggestions'].append('Many outliers - consider robust scaling')
        
        return analysis

    @classmethod
    def _analyze_datetime_column(cls, data: pd.Series, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze datetime columns."""
        analysis = {'type': 'datetime', 'subtype': 'datetime'}
        
        # Initialize issues and suggestions if not present
        if 'issues' not in analysis:
            analysis['issues'] = []
        if 'suggestions' not in analysis:
            analysis['suggestions'] = []
        
        try:
            datetime_data = pd.to_datetime(data)
            if 'stats' not in analysis:
                analysis['stats'] = {}
            analysis['stats'].update({
                'earliest': datetime_data.min(),
                'latest': datetime_data.max(),
                'span_days': (datetime_data.max() - datetime_data.min()).days
            })
            
            # Check for issues
            if analysis['stats']['span_days'] == 0:
                analysis['issues'].append('Single timestamp - no time span')
            
            # Generate suggestions
            analysis['suggestions'].extend([
                'Consider extracting time features (hour, day, month)',
                'Check for time series patterns'
            ])
            
        except Exception as e:
            analysis['issues'].append(f'Failed to parse as datetime: {e}')
        
        return analysis

    @classmethod
    def _analyze_object_column(cls, data: pd.Series, base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze object/string columns."""
        analysis = {'type': 'categorical', 'subtype': 'string'}
        
        # Initialize issues and suggestions if not present
        if 'issues' not in analysis:
            analysis['issues'] = []
        if 'suggestions' not in analysis:
            analysis['suggestions'] = []
        
        # Check if it's actually numeric stored as string
        try:
            numeric_converted = pd.to_numeric(data, errors='raise')
            analysis['type'] = 'numeric'
            analysis['subtype'] = 'string_numeric'
            analysis['issues'].append('Numeric data stored as strings')
            analysis['suggestions'].append('Convert to numeric type')
            return analysis
        except:
            pass
        
        # Analyze string patterns
        sample_values = data.dropna().head(100)
        if len(sample_values) > 0:
            # Check for common patterns
            if all(len(str(val)) <= 10 and str(val).isalpha() for val in sample_values):
                analysis['subtype'] = 'short_string'
            elif all('@' in str(val) for val in sample_values):
                analysis['subtype'] = 'email'
                analysis['suggestions'].append('Email format detected - consider encoding strategy')
            elif all(re.match(r'^\d{4}-\d{2}-\d{2}', str(val)) for val in sample_values):
                analysis['type'] = 'datetime'
                analysis['subtype'] = 'date_string'
                analysis['suggestions'].append('Date string format - convert to datetime')
        
        # Check cardinality
        if 'stats' in analysis and 'unique_percentage' in analysis['stats']:
            unique_ratio = analysis['stats']['unique_percentage']
            if unique_ratio < 10:
                analysis['subtype'] = 'low_cardinality'
                analysis['suggestions'].append('Low cardinality - good for one-hot encoding')
            elif unique_ratio > 90:
                analysis['subtype'] = 'high_cardinality'
                analysis['suggestions'].append('High cardinality - consider target encoding or embedding')
        
        # Generate suggestions
        if 'stats' in analysis and 'null_percentage' in analysis['stats'] and analysis['stats']['null_percentage'] > 0:
            analysis['suggestions'].append('Handle missing categorical values')
        
        return analysis

class SchemaDiscovery:
    """Main class for dataset schema discovery and preprocessing suggestions."""
    
    def __init__(self):
        self.role_detector = ColumnRoleDetector()
        self.type_analyzer = DataTypeAnalyzer()
    
    def discover_schema(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover the complete schema of a dataset.
        Args:
            df: Input DataFrame
            target_column: Optional target column name (if known)
        Returns:
            dict: Complete schema analysis
        """
        logging.info(f"Discovering schema for dataset with {len(df.columns)} columns...")
        
        schema = {
            'dataset_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'total_null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            },
            'columns': {},
            'suggested_targets': [],
            'preprocessing_suggestions': [],
            'data_quality_issues': []
        }
        
        # Analyze each column
        for column in df.columns:
            logging.info(f"Analyzing column: {column}")
            
            # Detect role
            role = self.role_detector.detect_column_role(column, df[column])
            
            # Analyze type
            type_analysis = self.type_analyzer.analyze_column_type(column, df[column])
            type_analysis['role'] = role
            
            # Override role if target column is specified
            if target_column and column == target_column:
                type_analysis['role'] = 'target'
            
            schema['columns'][column] = type_analysis
        
        # Generate high-level suggestions
        schema.update(self._generate_dataset_suggestions(schema))
        
        # Suggest potential targets
        if not target_column:
            schema['suggested_targets'] = self._suggest_target_columns(schema)
        
        logging.info("Schema discovery completed")
        return schema
    
    def _generate_dataset_suggestions(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level dataset preprocessing suggestions."""
        suggestions = []
        issues = []
        
        columns = schema['columns']
        
        # Check for data quality issues
        high_missing_cols = [col for col, info in columns.items() 
                           if 'stats' in info and 'null_percentage' in info['stats'] and info['stats']['null_percentage'] > 50]
        if high_missing_cols:
            issues.append(f"Columns with >50% missing values: {high_missing_cols}")
            suggestions.append("Consider dropping columns with excessive missing values")
        
        # Check for identifier columns
        id_cols = [col for col, info in columns.items() if info['role'] == 'identifier']
        if id_cols:
            suggestions.append(f"Identifier columns detected: {id_cols} - exclude from modeling")
        
        # Check for timestamp columns
        time_cols = [col for col, info in columns.items() if info['role'] == 'timestamp']
        if time_cols:
            suggestions.append(f"Timestamp columns detected: {time_cols} - consider time feature extraction")
        
        # Check data types
        numeric_cols = [col for col, info in columns.items() if info['type'] == 'numeric']
        categorical_cols = [col for col, info in columns.items() if info['type'] == 'categorical']
        
        if numeric_cols:
            suggestions.append(f"Numeric columns ({len(numeric_cols)}): Consider scaling/normalization")
        
        if categorical_cols:
            suggestions.append(f"Categorical columns ({len(categorical_cols)}): Consider encoding strategy")
        
        # Check for potential issues
        constant_cols = [col for col, info in columns.items() 
                        if 'issues' in info and ('No variance' in info['issues'] or 'Single value' in info['issues'])]
        if constant_cols:
            issues.append(f"Constant value columns: {constant_cols}")
            suggestions.append("Consider dropping constant value columns")
        
        return {
            'preprocessing_suggestions': suggestions,
            'data_quality_issues': issues
        }
    
    def _suggest_target_columns(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest potential target columns based on analysis."""
        suggestions = []
        columns = schema['columns']
        
        for col_name, col_info in columns.items():
            if col_info['role'] == 'identifier' or col_info['role'] == 'timestamp':
                continue
            
            score = 0
            reasons = []
            
            # Prefer columns with target-like names
            if col_info['role'] == 'target':
                score += 50
                reasons.append("Name suggests target variable")
            
            # Prefer categorical columns with moderate cardinality
            if col_info['type'] == 'categorical' and 'stats' in col_info and 2 <= col_info['stats'].get('unique_count', 0) <= 20:
                score += 30
                reasons.append("Good cardinality for classification")
            
            # Prefer numeric columns for regression
            if col_info['type'] == 'numeric' and 'stats' in col_info and col_info['stats'].get('unique_count', 0) > 10:
                score += 20
                reasons.append("Suitable for regression")
            
            # Penalize high missing values
            if 'stats' in col_info and col_info['stats'].get('null_percentage', 0) > 20:
                score -= 30
                reasons.append("High missing values")
            
            # Penalize too many unique values for categorical
            if col_info['type'] == 'categorical' and 'stats' in col_info and col_info['stats'].get('unique_percentage', 0) > 90:
                score -= 20
                reasons.append("Too many unique values")
            
            if score > 0:
                suggestions.append({
                    'column': col_name,
                    'score': score,
                    'reasons': reasons,
                    'suggested_task': 'classification' if col_info['type'] == 'categorical' else 'regression'
                })
        
        # Sort by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions

def discover_dataset_schema(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to discover dataset schema.
    Args:
        df: Input DataFrame
        target_column: Optional target column name
    Returns:
        dict: Schema analysis results
    """
    discovery = SchemaDiscovery()
    return discovery.discover_schema(df, target_column)

if __name__ == "__main__":
    # Test the schema discovery
    print("Testing Schema Discovery...")
    
    # Create test data
    test_data = pd.DataFrame({
        'machine_id': ['M001', 'M002', 'M003', 'M004', 'M005'],
        'timestamp': pd.date_range('2023-01-01', periods=5),
        'temperature': [300.1, 301.5, None, 302.1, 301.9],
        'pressure': [100, 200, 300, 400, 500],
        'status': ['Normal', 'Warning', 'Normal', 'Critical', 'Normal'],
        'maintenance_priority': [1, 2, 1, 3, 1],
        'operator_name': ['John', 'Jane', 'John', 'Jane', 'John']
    })
    
    print("Test data:")
    print(test_data)
    print()
    
    # Discover schema
    schema = discover_dataset_schema(test_data)
    
    print("Schema Discovery Results:")
    print(f"Dataset shape: {schema['dataset_info']['shape']}")
    print(f"Memory usage: {schema['dataset_info']['memory_usage_mb']:.2f} MB")
    print()
    
    print("Column Analysis:")
    for col_name, col_info in schema['columns'].items():
        print(f"  {col_name}:")
        print(f"    Role: {col_info['role']}")
        print(f"    Type: {col_info['type']} ({col_info['subtype']})")
        print(f"    Issues: {col_info['issues']}")
        print(f"    Suggestions: {col_info['suggestions']}")
        print()
    
    print("Suggested Targets:")
    for suggestion in schema['suggested_targets']:
        print(f"  {suggestion['column']} (score: {suggestion['score']}): {suggestion['suggested_task']}")
        print(f"    Reasons: {', '.join(suggestion['reasons'])}")
    
    print("\nPreprocessing Suggestions:")
    for suggestion in schema['preprocessing_suggestions']:
        print(f"  - {suggestion}")
    
    print("\nData Quality Issues:")
    for issue in schema['data_quality_issues']:
        print(f"  - {issue}")
