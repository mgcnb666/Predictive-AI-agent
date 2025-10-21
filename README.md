# 🎯 Prediction AI Agent

[English](README.md) | [中文](README_CN.md)

> 🌐 **Universal Prediction System** - Multi-domain predictions for sports, weather, elections, and more

An intelligent multi-domain prediction system based on [OpenDeepSearch](https://github.com/sentient-agi/OpenDeepSearch)


## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd prediction-ai-agent
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `env.example.txt` to create `.env` file:

```bash
cp env.example.txt .env
```

Edit `.env` file and fill in your API keys:

```env
# Required API keys
SERPER_API_KEY=your-serper-api-key
JINA_API_KEY=your-jina-api-key
OPENROUTER_API_KEY=your-openrouter-api-key

# LLM model configuration (using Gemini 2.0 Flash)
LITELLM_MODEL_ID=openrouter/google/gemini-2.0-flash-001
OPENROUTER_MODEL=google/gemini-2.0-flash-001

# Search configuration
SEARCH_PROVIDER=serper
RERANKER=jina
```

### 3. Usage Methods

#### Method A: Chat Mode (Recommended) 🆕

**The simplest way - ask like chatting!**

```bash
# Start chat
python3 chat.py

python3 main.py api --host 0.0.0.0 --port 8789

**Supported Commands:**
- `/help` - Show help
- `/history` - View conversation history
- `/context` - View current context
- `/clear` - Clear context
- `/set <key> <value>` - Set preference
- `/quit` or `/exit` - Exit

