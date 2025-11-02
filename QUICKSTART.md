# Quick Start Guide

Get up and running with Smart Manufacturing MAS in 5 minutes!

## 🎯 5-Minute Setup

### Step 1: Installation (2 minutes)

```bash
# Clone repository
git clone <repository-url>
cd smart_manufacturing_mas_code

# Create and activate virtual environment
python3 -m venv mas_venv
source mas_venv/bin/activate  # On Windows: mas_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Choose Your LLM (1 minute)

**Option A: Google Gemini (Cloud)**
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

**Option B: Ollama (Local)**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:4b
```

**Option C: No setup needed!**
Just skip API keys and use `--planner-llm mock` (limited functionality)

### Step 3: Run Your First Analysis (2 minutes)

**Interactive Mode** (Guided):
```bash
python3 main_llm.py
```

**Auto Mode** (Faster):
```bash
python3 main_llm.py --auto --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
```

**With Local LLM**:
```bash
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b --auto
```

## 📊 What You'll See

```
================================================================================
🧠 INTELLIGENT MULTI-AGENT SYSTEM FOR SMART MANUFACTURING
================================================================================

[INFO] - Loading dataset...
[INFO] - Preprocessing complete. Shape: (1430, 9)
[INFO] - Analysis complete. Model: RandomForestClassifier, Accuracy: 0.87
[INFO] - Generated 50 recommendations

================================================================================
INTELLIGENT SUMMARY
================================================================================

✅ Workflow completed successfully in 3.2 seconds
📊 Model Performance: RandomForestClassifier - 87% accuracy
🔧 Generated 50 prescriptive recommendations
🎯 Top Priority: 15 critical maintenance actions identified
```

## 🎮 Common Commands

### Basic Operations

```bash
# List available datasets
ls data/*/

# Run with auto mode
python3 main_llm.py --auto

# Process all datasets
python3 main_llm.py --batch

# Debug mode
export LOG_LEVEL=DEBUG && python3 main_llm.py --auto
```

### LLM Options

```bash
# Use local LLM only
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b

# Use Gemini with local models for decisions
python3 main_llm.py --planner-llm gemini --decision-llm ollama --decision-model llama3:8b

# Use mock mode (no LLM required)
python3 main_llm.py --planner-llm mock --decision-llm mock
```

## 📁 Output Files

After running, check the `logs/` directory:

```bash
# View latest workflow report
ls -lt logs/workflow_report_*.json | head -1

# View recommendations
ls -lt logs/detailed_results_*.json | head -1

# View audit trail
cat logs/hitl_audit.json
```

## 🐛 Quick Troubleshooting

**Problem**: `ModuleNotFoundError`
```bash
source mas_venv/bin/activate
pip install -r requirements.txt
```

**Problem**: `API key not configured`
```bash
# Option 1: Use local LLM
python3 main_llm.py --decision-llm ollama --decision-model qwen3:4b

# Option 2: Add Gemini API key
echo "GEMINI_API_KEY=your_key" > .env
```

**Problem**: `Ollama connection failed`
```bash
ollama serve
ollama pull qwen3:4b
ollama list  # Verify model is available
```

**Problem**: `Dataset not found`
```bash
# Use absolute path
python3 main_llm.py --auto --dataset "$(pwd)/data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv"
```

## 📚 Next Steps

- **Learn More**: Read [Detailed Usage Guide](documentation/usage_guide.md)
- **Understand Architecture**: See [Architecture Documentation](documentation/architecture_and_workflow.md)
- **Explore Features**: Check [Adaptive Intelligence System](documentation/adaptive_intelligence_system.md)

## 💡 Pro Tips

1. **Start with auto mode** to understand the workflow quickly
2. **Use local LLMs** for faster, offline operation
3. **Enable debug mode** when troubleshooting: `export LOG_LEVEL=DEBUG`
4. **Check logs** for detailed information about decisions and performance
5. **Try batch mode** to process multiple datasets at once

## ❓ Need Help?

- Check [Troubleshooting Guide](documentation/usage_guide.md#troubleshooting)
- Review example outputs in `logs/` directory
- Enable debug mode: `export LOG_LEVEL=DEBUG`
- Check all command options: `python3 main_llm.py --help`

---

**Ready?** Run `python3 main_llm.py` to start your first analysis! 🚀

