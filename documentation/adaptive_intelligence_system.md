# 🧠 Adaptive Intelligence System - Technical Documentation

## Overview

The Adaptive Intelligence System is a sophisticated component that automatically detects poor model performance and intelligently switches to alternative models to achieve better results. This system operates transparently and requires no manual intervention, making the MAS framework more robust and intelligent.

## System Architecture

### Core Components

1. **Performance Monitor**: Continuously monitors model performance metrics
2. **Model Diversity Engine**: Manages multiple model candidates for each task type
3. **Adaptive Controller**: Orchestrates the model switching process
4. **Performance Tracker**: Records and compares model performance
5. **Result Selector**: Chooses the best performing model

### Integration Points

- **DynamicAnalysisAgent**: Primary integration point for model execution
- **LLMPlannerAgent**: Receives performance feedback and context
- **Reporting System**: Logs adaptive intelligence activities

## Activation Logic

### Performance Threshold Detection

The system uses specific thresholds to determine when to activate adaptive intelligence:

#### Regression Tasks
```python
def _is_poor_performance(self, results: Dict[str, Any]) -> bool:
    if self.problem_type == "regression":
        r2 = results.get("r2", -float('inf'))
        return r2 < 0.1  # Poor R² score
```

**Threshold**: R² < 0.1 (10% variance explained)
**Rationale**: Indicates the model performs worse than a simple baseline model

#### Classification Tasks
```python
elif self.problem_type == "classification":
    accuracy = results.get("accuracy", 0)
    return accuracy < 0.6  # Poor accuracy
```

**Threshold**: Accuracy < 0.6 (60% correct predictions)
**Rationale**: Indicates poor classification performance

#### Anomaly Detection
```python
else:  # anomaly_detection
    return False  # No clear performance metric for anomaly detection
```

**Threshold**: No automatic activation
**Rationale**: Performance metrics vary significantly by use case

## Model Diversity Strategy

### Regression Models (Order of Testing)

1. **RandomForestRegressor**
   - **Purpose**: Robust ensemble method for complex patterns
   - **Strengths**: Handles non-linear relationships, feature interactions
   - **Use Case**: Complex manufacturing data with multiple variables

2. **LinearRegression**
   - **Purpose**: Simple linear relationship modeling
   - **Strengths**: Fast, interpretable, good baseline
   - **Use Case**: Linear relationships in sensor data

3. **Ridge**
   - **Purpose**: Regularized linear regression
   - **Strengths**: Handles multicollinearity, prevents overfitting
   - **Use Case**: High-dimensional data with correlated features

4. **Lasso**
   - **Purpose**: Feature selection with regularization
   - **Strengths**: Automatic feature selection, sparse solutions
   - **Use Case**: Data with many irrelevant features

5. **SVR**
   - **Purpose**: Support Vector Regression for non-linear patterns
   - **Strengths**: Handles non-linear relationships, robust to outliers
   - **Use Case**: Complex non-linear manufacturing processes

### Classification Models (Order of Testing)

1. **RandomForestClassifier**
   - **Purpose**: Robust ensemble method for classification
   - **Strengths**: Handles non-linear boundaries, feature interactions
   - **Use Case**: Complex classification problems

2. **LogisticRegression**
   - **Purpose**: Linear classification with regularization
   - **Strengths**: Fast, interpretable, probabilistic outputs
   - **Use Case**: Linear decision boundaries

3. **SVC**
   - **Purpose**: Support Vector Classification for complex boundaries
   - **Strengths**: Handles non-linear boundaries, high-dimensional data
   - **Use Case**: Complex classification with non-linear patterns

## Implementation Details

### Core Algorithm

```python
def _try_multiple_models(self) -> Optional[Dict[str, Any]]:
    """
    Try multiple models and return the best performing one.
    """
    logging.info("🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...")
    
    # Define model candidates based on task type
    if self.task == "classification":
        model_candidates = [
            ("RandomForestClassifier", self._run_random_forest),
            ("LogisticRegression", self._run_logistic_regression),
            ("SVC", self._run_svc)
        ]
    elif self.task == "regression":
        model_candidates = [
            ("RandomForestRegressor", self._run_random_forest_regressor),
            ("LinearRegression", self._run_linear_regression),
            ("Ridge", self._run_ridge),
            ("Lasso", self._run_lasso),
            ("SVR", self._run_svr)
        ]
    else:
        return None
    
    best_performance = -float('inf')
    best_model_name = None
    best_results = None
    
    for model_name, model_func in model_candidates:
        if model_name in self.tried_models:
            logging.info(f"⏭️ Skipping {model_name} (already tried)")
            continue
            
        try:
            logging.info(f"🔄 Trying {model_name}...")
            results = model_func()
            
            if results:
                # Calculate performance metric
                if self.task == "classification":
                    performance = results.get("accuracy", 0)
                else:  # regression
                    performance = results.get("r2_score", -float('inf'))
                
                logging.info(f"   {model_name} performance: {performance:.4f}")
                
                if performance > best_performance:
                    best_performance = performance
                    best_model_name = model_name
                    best_results = results
                    self.best_model = model_name
                    self.best_results = results
                    self.best_performance = performance
            
            self.tried_models.append(model_name)
            
        except Exception as e:
            logging.warning(f"   {model_name} failed: {str(e)}")
            self.tried_models.append(model_name)
            continue
    
    if best_results:
        logging.info(f"🏆 Best model: {best_model_name} (performance: {best_performance:.4f})")
        return best_results
    else:
        logging.error("❌ All models failed")
        return None
```

### Integration with Analysis Workflow

```python
def _execute_analysis_step(self):
    # ... existing code ...
    
    agent = DynamicAnalysisAgent(self.preprocessed_data, self.target_column, task=self.problem_type, params=params)
    results = agent.run()
    
    # Check if performance is poor and try adaptive intelligence
    if results and self._is_poor_performance(results):
        logging.info("🧠 Poor performance detected! Trying adaptive intelligence with multiple models...")
        results = agent.run(force_retry=True)
    
    # ... rest of the method ...
```

## Performance Tracking

### State Variables

```python
class DynamicAnalysisAgent:
    def __init__(self, ...):
        # ... existing initialization ...
        self.tried_models = []  # Track models already tried
        self.best_performance = -float('inf')  # Track best performance
        self.best_model = None
        self.best_results = None
```

### Performance Metrics

- **Tried Models**: List of models already attempted
- **Best Performance**: Highest performance score achieved
- **Best Model**: Name of the best performing model
- **Best Results**: Complete results from the best model

## Logging and Monitoring

### Activation Logs

When adaptive intelligence activates, the system logs:

```
[INFO] - 🧠 Poor performance detected! Trying adaptive intelligence with multiple models...
[INFO] - 🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...
[INFO] - 🔄 Trying RandomForestRegressor...
[INFO] -    RandomForestRegressor performance: 0.2341
[INFO] - 🔄 Trying LinearRegression...
[INFO] -    LinearRegression performance: 0.4567
[INFO] - 🔄 Trying Ridge...
[INFO] -    Ridge performance: 0.5234
[INFO] - 🏆 Best model: Ridge (performance: 0.5234)
```

### Error Handling Logs

```
[WARNING] -    SVR failed: ConvergenceWarning: Liblinear failed to converge
[INFO] - ⏭️ Skipping RandomForestRegressor (already tried)
[ERROR] - ❌ All models failed
```

## LLM Integration

### Prompt Enhancement

The LLM is informed about adaptive intelligence capabilities through enhanced prompts:

```
- ADAPTIVE INTELLIGENCE: The system will automatically try multiple models if performance is poor
- If you see 'ADAPTIVE INTELLIGENCE' in logs, the system is trying different models for better performance
```

### Context Awareness

The LLM understands:
- When adaptive intelligence is active
- What to expect when performance is poor
- How to interpret adaptive intelligence logs

## Error Handling

### Model Failures

```python
try:
    logging.info(f"🔄 Trying {model_name}...")
    results = model_func()
    # ... process results ...
except Exception as e:
    logging.warning(f"   {model_name} failed: {str(e)}")
    self.tried_models.append(model_name)
    continue
```

### All Models Fail

```python
if best_results:
    logging.info(f"🏆 Best model: {best_model_name} (performance: {best_performance:.4f})")
    return best_results
else:
    logging.error("❌ All models failed")
    return None
```

### Memory Management

- **Tried Models Tracking**: Prevents infinite loops
- **Performance Validation**: Ensures metrics are valid before comparison
- **Graceful Degradation**: Returns original results if all models fail

## Benefits

1. **Automatic Optimization**: No manual model selection required
2. **Better Performance**: System finds the best model for each dataset
3. **Transparent Process**: Clear logging of all model attempts
4. **Intelligent Fallback**: Graceful handling of poor initial performance
5. **Learning Capability**: System builds knowledge of what works for different data types

## Future Enhancements

### Phase 3: Learning System
- **Pattern Recognition**: Learn which models work best for different data characteristics
- **Performance Prediction**: Predict model performance before training
- **Adaptive Thresholds**: Adjust performance thresholds based on historical data
- **Ensemble Methods**: Combine multiple models for even better performance

### Advanced Features
- **Hyperparameter Optimization**: Automatically tune model parameters
- **Feature Engineering**: Intelligent feature creation based on data characteristics
- **Cross-Validation**: More robust performance evaluation
- **Model Explanation**: SHAP values and feature importance analysis

---

*This technical documentation provides a comprehensive understanding of the Adaptive Intelligence System implementation and operation.*
