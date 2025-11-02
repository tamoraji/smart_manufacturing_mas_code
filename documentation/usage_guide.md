# Intelligent MAS Usage and Workflow Guide

This document provides comprehensive instructions for using the Intelligent Multi-Agent System (MAS) for smart manufacturing. The system offers both traditional rule-based workflows and advanced LLM-powered intelligent workflows.

## Installation

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd smart_manufacturing_mas_code

# Create virtual environment
python3 -m venv mas_venv

# Activate virtual environment
# On macOS/Linux:
source mas_venv/bin/activate
# On Windows:
mas_venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

**Required packages:**
- `google-generativeai` - Google Gemini API
- `python-dotenv` - Environment variable management
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms

### Step 3: Configure API Keys (Optional)

If using Google Gemini, create a `.env` file:

```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

**Note**: You can skip this step if using local LLMs (Ollama) exclusively.

### Step 4: Setup Local LLM (Optional)

For offline operation, install and configure Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen3:4b

# Verify installation
ollama list
```

### Step 5: Verify Installation

Test your installation:

```bash
python3 -c "import pandas as pd; import sklearn; print('✓ Installation successful')"
```

## Quick Start

### Prerequisites

✅ **Completed Installation Steps**
1. Python Environment: Python 3.8+ with virtual environment
2. API Keys: Google Gemini API key (optional if using local LLMs)
3. Dependencies: Installed via `pip install -r requirements.txt`

### Basic Usage

```bash
# Activate virtual environment
source mas_venv/bin/activate  # On Windows: mas_venv\Scripts\activate

# Run with default dataset (interactive mode)
python3 main_llm.py

# Run with specific dataset (auto mode)
python3 main_llm.py --auto --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"

# Run with local LLM
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b
```

## Usage Modes

### 1. **Interactive Mode** (Default)
Guided setup with human oversight and decision-making.

```bash
python3 main_llm.py
```

**Features:**
- Step-by-step dataset selection
- Manual target column and problem type selection
- Human approval for critical decisions
- Detailed explanations of each step

### 2. **Auto Mode** (`--auto`)
Fully automated workflow with intelligent schema discovery.

```bash
python3 main_llm.py --auto --dataset "path/to/dataset.csv"
```

**Features:**
- Automatic schema discovery and analysis
- Intelligent target column detection
- Problem type classification (classification/regression/anomaly detection)
- Minimal human intervention

### 3. **Batch Mode** (`--batch`)
Process multiple datasets automatically.

```bash
python3 main_llm.py --batch
```

**Features:**
- Processes all datasets in `data/` directory
- Comparative analysis across datasets
- Automated reporting and recommendations

## Command Line Options

### Core Options
- `--dataset PATH`: Specify dataset path
- `--auto`: Enable auto mode with schema discovery
- `--batch`: Process all datasets in data directory
- `--interface {cli,web}`: Choose interface type (default: cli)

### LLM Configuration
- `--planner-llm {gemini,mock}`: Planner LLM backend (default: gemini)
- `--planner-model MODEL`: Planner model name
- `--decision-llm {ollama,mock}`: Decision LLM backend
- `--decision-model MODEL`: Decision model name (e.g., qwen3:4b, llama3:8b)

### Examples

```bash
# Use local LLM for all decisions
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b --auto

# Process specific dataset with custom LLM
python3 main_llm.py --dataset "data/custom_dataset.csv" --planner-llm gemini --decision-llm ollama

# Batch process with local models
python3 main_llm.py --batch --decision-llm ollama --decision-model llama3:8b
```

## Agent Architecture

### 1. **LLM Planner Agent** (`llm_planner_agent.py`)
- **Role**: High-level workflow orchestration using Google Gemini
- **Capabilities**: 
  - Dynamic tool selection based on context
  - Chain-of-thought reasoning
  - Error recovery and adaptation
  - Human-in-the-loop interactions

### 2. **Data Loader Agent** (`data_loader_agent.py`)
- **Role**: Dataset loading and initial inspection
- **Capabilities**:
  - Comprehensive data quality assessment
  - Metadata extraction and analysis
  - Missing value detection
  - Statistical summary generation

### 3. **Intelligent Preprocessing Agent** (`preprocessing_agent.py`)
- **Role**: Advanced data preprocessing with intelligence
- **Capabilities**:
  - Intelligent feature analysis (correlation, importance, redundancy)
  - Automatic categorical encoding strategy selection
  - High-cardinality feature handling
  - Data leakage prevention
  - Domain-aware preprocessing

### 4. **Dynamic Analysis Agent** (`dynamic_analysis_agent.py`)
- **Role**: Multi-model analysis with intelligent selection
- **Capabilities**:
  - Automatic model family selection
  - Hyperparameter optimization
  - Performance evaluation with multiple metrics
  - Cross-validation and robust testing

### 5. **Optimization Agent** (`optimization_agent.py`)
- **Role**: Prescriptive recommendation generation
- **Capabilities**:
  - Cost-benefit analysis
  - Priority ranking
  - Actionable maintenance recommendations
  - Resource allocation optimization

### 6. **Schema Discovery System** (`utils/schema_discovery.py`)
- **Role**: Automatic dataset understanding
- **Capabilities**:
  - Column role detection (identifier, target, feature, timestamp)
  - Data type analysis and classification
  - Preprocessing suggestion generation
  - Data quality issue identification

### 7. **Tool Decider** (`utils/tool_decider.py`)
- **Role**: Intelligent tool selection
- **Capabilities**:
  - Rule-based decision making
  - LLM-powered tool selection
  - Hybrid decision strategies
  - Performance-based adaptation

## Advanced Features

### **Intelligent Feature Analysis**
The system automatically performs comprehensive feature analysis:

- **Correlation Analysis**: Identifies highly correlated features
- **Feature Importance**: Ranks features by predictive power
- **Mutual Information**: Measures feature-target relationships
- **Redundancy Detection**: Identifies and removes redundant features
- **High-Cardinality Handling**: Manages categorical features with many unique values

### **Multi-Model Approach**
- **Automatic Model Selection**: Chooses appropriate models based on data characteristics
- **Ensemble Methods**: Combines multiple models for improved performance
- **Performance Tracking**: Monitors and adapts based on results
- **Cross-Validation**: Robust evaluation with multiple metrics

### **Data Quality Assurance**
- **Comprehensive Validation**: Checks for data quality issues
- **Missing Value Analysis**: Identifies and handles missing data
- **Outlier Detection**: Finds and manages unusual values
- **Data Leakage Prevention**: Ensures proper feature-target separation

## Output and Reporting

### **Console Output**
The system provides detailed console output including:
- Step-by-step progress updates
- Intelligent feature analysis results
- Model performance metrics
- Recommendation summaries
- Error messages and recovery actions

### **Logging**
Comprehensive logging to `logs/` directory:
- `hitl_audit.json`: Human-in-the-loop decision audit trail
- Console logs: Detailed execution information
- Error logs: Debugging and troubleshooting information

### **Results Summary**
Each run generates:
- **Performance Metrics**: R², Accuracy, MSE, etc.
- **Feature Analysis**: Importance rankings and recommendations
- **Maintenance Recommendations**: Prioritized action items
- **Model Confidence**: Reliability assessment and warnings

## 🧠 Adaptive Intelligence System

### **How It Works**

The system automatically detects poor model performance and intelligently switches to alternative models to achieve better results. This happens transparently without any manual intervention.

**Important**: All detailed logs are preserved for debugging and verification. The intelligent summarization provides a clean summary at the end while maintaining full logging transparency.

### **When Adaptive Intelligence Activates**

#### **Regression Tasks:**
- **Trigger**: R² Score < 0.1 (10% variance explained)
- **Action**: System tries 5 different regression models
- **Models**: RandomForest, LinearRegression, Ridge, Lasso, SVR

#### **Classification Tasks:**
- **Trigger**: Accuracy < 0.6 (60% correct predictions)
- **Action**: System tries 3 different classification models
- **Models**: RandomForest, LogisticRegression, SVC

### **What You'll See in Logs**

When adaptive intelligence activates, you'll see logs like:
```
[INFO] - 🧠 Poor performance detected! Trying adaptive intelligence with multiple models...
[INFO] - 🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...
[INFO] - 🔄 Trying RandomForestRegressor...
[INFO] -    RandomForestRegressor performance: 0.2341
[INFO] - 🔄 Trying LinearRegression...
[INFO] -    LinearRegression performance: 0.4567
[INFO] - 🏆 Best model: LinearRegression (performance: 0.4567)
```

### **Benefits**

1. **Automatic Optimization**: No need to manually try different models
2. **Better Results**: System finds the best model for your data
3. **Transparent Process**: Clear logging of all model attempts
4. **Intelligent Fallback**: Handles poor initial performance gracefully

### **Performance Thresholds**

| Task Type | Threshold | Meaning |
|-----------|-----------|---------|
| Regression | R² < 0.1 | Model explains less than 10% of variance |
| Classification | Accuracy < 0.6 | Model predicts correctly less than 60% of time |
| Anomaly Detection | N/A | No automatic activation (varies by use case) |

## 🧠 Intelligent Summarization System

### **Overview**

The system now includes an intelligent summarization feature that provides clean, professional summaries at the end of workflow execution while preserving all detailed logs for debugging and verification.

### **What You Get**

1. **Complete Logging**: All detailed logs are preserved exactly as before
2. **Intelligent Summary**: LLM-generated clean summary at the end
3. **Structured Data**: All results stored in organized JSON format
4. **Professional Output**: Stakeholder-ready summaries

### **Log Structure**

```
[INFO] - [Detailed workflow logs...]  # All existing logs preserved
[INFO] - [Summarizer] Workflow started: Dataset: smart_maintenance_dataset.csv
[INFO] - [Summarizer] Model: RandomForestRegressor: Performance: R²: 0.8933
[INFO] - [Summarizer] Recommendations: Generated prescriptive recommendations
[INFO] - [Summarizer] Workflow Complete: All steps completed successfully

================================================================================
🧠 GENERATING INTELLIGENT SUMMARY...
================================================================================

[Clean, professional summary generated by LLM...]
```

### **Output Files**

- **Console**: All detailed logs + intelligent summary
- **`logs/workflow_report_*.json`**: Detailed workflow report
- **`logs/detailed_results_*.json`**: Complete structured data

## Troubleshooting

### **Common Issues**

#### 1. **Import or Module Errors**

**Problem**: `ModuleNotFoundError` or import issues

**Solution**:
```bash
# Verify virtual environment is activated
which python  # Should point to mas_venv/bin/python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### 2. **API Key Missing**

**Problem**: `API key not configured` or Gemini API errors

**Solution**:
```bash
# Set Gemini API key in .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Verify .env file exists
cat .env

# Alternative: Use local LLM instead
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b
```

#### 3. **Local LLM Connection Issues**

**Problem**: Ollama connection failed or model not found

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull required model
ollama pull qwen3:4b

# Test connection
ollama run qwen3:4b "Hello"
```

#### 4. **Dataset Not Found**

**Problem**: File not found errors

**Solution**:
```bash
# Check dataset path
ls -la "data/Smart Manufacturing Maintenance Dataset/"

# Use absolute path
python3 main_llm.py --auto --dataset "$(pwd)/data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"

# Verify CSV format
head -5 "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
```

#### 5. **Memory Issues**

**Problem**: Out of memory errors

**Solution**:
```bash
# Use smaller LLM models
python3 main_llm.py --decision-model qwen3:4b

# Use auto mode to skip interactive prompts
python3 main_llm.py --auto

# Check system resources
free -h  # Linux
vm_stat  # macOS
```

#### 6. **ValueError: could not convert string to float**

**Problem**: Data type conversion errors during preprocessing

**Solution**: This should be automatically handled. If it occurs:
```bash
# Check for ID columns in your dataset
head -5 your_dataset.csv

# Make sure dataset has proper column types
# String identifiers should be preserved automatically
```

#### 7. **Performance Issues**

**Problem**: Slow execution or high resource usage

**Solution**:
```bash
# Use local LLM for faster inference
python3 main_llm.py --decision-llm ollama

# Use mock mode for testing
python3 main_llm.py --planner-llm mock --decision-llm mock

# Reduce dataset size for testing
python3 main_llm.py --dataset "small_test_dataset.csv"
```

### **Debug Mode**
Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python3 main_llm.py --auto
```

## Performance Optimization

### **For Large Datasets**
- Use auto mode for faster processing
- Consider using local LLMs for better performance
- Enable batch processing for multiple datasets

### **For Better Accuracy**
- Use interactive mode for manual oversight
- Enable intelligent feature analysis
- Use ensemble methods and multiple models

### **For Production Use**
- Use auto mode with schema discovery
- Enable comprehensive logging
- Set up monitoring and alerting

## Next Steps

### **Phase 1: Enhanced Reasoning** (Current)
- Chain-of-thought reasoning for complex decisions
- Context-aware planning with memory
- Intelligent error recovery strategies

### **Phase 2: Adaptive Intelligence** (Coming Soon)
- Dynamic workflow adaptation
- Multi-model orchestration
- Advanced domain knowledge integration

### **Phase 3: Learning System** (Future)
- Self-improving capabilities
- Predictive performance optimization
- Advanced manufacturing intelligence

---

*This guide covers the current capabilities of the Intelligent MAS framework. The system continues to evolve with enhanced reasoning and adaptive intelligence features.*
