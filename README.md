# Smart Manufacturing Multi-Agent System (MAS)

An intelligent Multi-Agent System for predictive maintenance and optimization in smart manufacturing. This system uses LLM-powered orchestration to automatically load, preprocess, analyze, and generate prescriptive recommendations from manufacturing data.

## 🚀 Features

- **LLM-Powered Orchestration**: Intelligent workflow planning using Google Gemini or local LLMs (Ollama)
- **Adaptive Intelligence**: Automatic model selection and performance optimization
- **Intelligent Preprocessing**: Automatic feature analysis, encoding, and data quality handling
- **Multi-Model Analysis**: Supports classification, regression, and anomaly detection
- **Prescriptive Recommendations**: Actionable maintenance suggestions with priority ranking
- **Human-in-the-Loop**: Interactive approval workflow for critical decisions
- **Comprehensive Logging**: Detailed audit trails and performance metrics

## ✨ Recent Improvements

### Bug Fixes & Enhancements (Latest Release)

1. **Fixed One-Hot Encoding Issue**: Protected identifier columns (e.g., Machine_ID) are now properly handled as pass-through features instead of being one-hot encoded, reducing feature explosion and improving performance.

2. **Fixed ID Column Handling**: All models now automatically drop ID columns during training while preserving them for recommendations and reporting.

3. **Enhanced Contributing Factors**: All priority levels (Critical, Medium, Low) now show detailed contributing factors based on actual feature values instead of generic messages.

4. **Improved Workflow Completion**: Fixed LLM validation logic to properly handle workflow completion signals.

5. **Better Data Flow Logic**: Improved tracking of data shapes through the entire pipeline for better debugging and transparency.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **API Key**: Google Gemini API key (optional if using local LLMs)
- **Ollama**: Optional, for local LLM support

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd smart_manufacturing_mas_code
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv mas_venv

# Activate virtual environment
# On macOS/Linux:
source mas_venv/bin/activate
# On Windows:
mas_venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (Optional)

If using Google Gemini, create a `.env` file:

```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

### 5. Install Ollama (Optional - for Local LLMs)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (example with Qwen3)
ollama pull qwen3:4b
```

## 🎯 Quick Start

### Interactive Mode (Recommended for First-Time Users)

```bash
python3 main_llm.py
```

This will guide you through:
- Dataset selection
- Feature and target selection
- Problem type identification
- Approval of critical decisions

### Automated Mode

```bash
python3 main_llm.py --auto --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
```

### With Local LLM

```bash
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b
```

## 📖 Usage Examples

### Basic Analysis

```bash
# Run with default Gemini LLM
python3 main_llm.py

# Run with auto schema discovery
python3 main_llm.py --auto --dataset "path/to/dataset.csv"

# Run with local LLM
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b

# Batch process all datasets
python3 main_llm.py --batch
```

### Advanced Configuration

```bash
# Custom LLM configuration
python3 main_llm.py \
    --planner-llm gemini \
    --decision-llm ollama \
    --decision-model llama3:8b \
    --auto

# Specific dataset with auto mode
python3 main_llm.py \
    --auto \
    --dataset "data/Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv"
```

## 🧠 Architecture

### Agents

1. **LLM Planner Agent**: Orchestrates workflow using LLM reasoning
2. **Data Loader Agent**: Loads and inspects datasets
3. **Preprocessing Agent**: Intelligent feature analysis and data preparation
4. **Dynamic Analysis Agent**: Multi-model analysis with automatic selection
5. **Optimization Agent**: Generates prescriptive recommendations

### Key Components

- **Schema Discovery**: Automatic dataset understanding
- **Tool Decider**: Intelligent preprocessing/model selection
- **Adaptive Intelligence**: Performance-based model switching
- **Intelligent Summarization**: Clean output with full logging

## 📊 Supported Problem Types

- **Classification**: Predict categorical outcomes (e.g., maintenance priority)
- **Regression**: Predict continuous values (e.g., failure probability)
- **Anomaly Detection**: Identify unusual patterns (no target needed)

## 🔧 Configuration

### Command Line Options

| Option | Description |
|--------|-------------|
| `--planner-llm` | LLM backend for planner (gemini, ollama, mock) |
| `--planner-model` | Model name for planner |
| `--decision-llm` | LLM backend for decisions (ollama, mock, or None) |
| `--decision-model` | Model name for decisions (e.g., qwen3:4b) |
| `--dataset` | Path to CSV dataset |
| `--auto` | Enable auto mode with schema discovery |
| `--batch` | Process all datasets in data/ directory |
| `--interface` | HITL interface (cli or web) |

## 📁 Project Structure

```
smart_manufacturing_mas_code/
├── agents/                    # Core agent implementations
│   ├── llm_planner_agent.py   # LLM orchestration
│   ├── data_loader_agent.py   # Data loading
│   ├── preprocessing_agent.py # Preprocessing
│   ├── dynamic_analysis_agent.py # Analysis
│   └── optimization_agent.py  # Recommendations
├── utils/                     # Utility modules
│   ├── schema_discovery.py    # Auto schema detection
│   ├── tool_decider.py        # Model/tool selection
│   ├── hitl_interface.py      # Human-in-loop UI
│   └── reporting.py           # Logging and reporting
├── data/                      # Sample datasets
│   ├── Smart Manufacturing Maintenance Dataset/
│   └── Intelligent Manufacturing Dataset/
├── documentation/             # Documentation
│   ├── usage_guide.md         # Detailed usage guide
│   ├── architecture_and_workflow.md
│   └── adaptive_intelligence_system.md
├── logs/                      # Generated logs
├── main_llm.py               # Main entry point
└── requirements.txt          # Dependencies
```

## 📈 Output

### Console Output

The system provides real-time progress updates and a final intelligent summary.

### Log Files

All runs generate:
- `logs/workflow_report_*.json`: Detailed workflow report
- `logs/detailed_results_*.json`: Complete structured data
- `logs/hitl_audit.json`: Human approval audit trail

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure virtual environment is activated
source mas_venv/bin/activate

# Verify installation
pip list | grep scikit-learn
```

**2. API Key Issues**
```bash
# Check .env file exists
cat .env

# Verify API key is valid
python -c "import google.generativeai as genai; genai.configure(api_key='your_key')"
```

**3. Ollama Connection Issues**
```bash
# Start Ollama service
ollama serve

# Test model availability
ollama list
```

**4. Dataset Issues**
```bash
# Verify dataset path
ls -la data/

# Check CSV format
head -5 data/Smart\ Manufacturing\ Maintenance\ Dataset/smart_maintenance_dataset.csv
```

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python3 main_llm.py --auto
```

## 🎓 Learn More

- [Detailed Usage Guide](documentation/usage_guide.md)
- [Architecture and Workflow](documentation/architecture_and_workflow.md)
- [Adaptive Intelligence System](documentation/adaptive_intelligence_system.md)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

This project is developed for research in intelligent manufacturing and predictive maintenance systems.

## 🔗 Contact

[Add contact information]

---

**Ready to get started?** Run `python3 main_llm.py` to begin your first analysis!

