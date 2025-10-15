# 🎯 Prediction AI Agent

[English](README.md) | [中文](README_CN.md)

> 🌐 **Universal Prediction System** - Multi-domain predictions for sports, weather, elections, and more

An intelligent multi-domain prediction system based on [OpenDeepSearch](https://github.com/sentient-agi/OpenDeepSearch) 


## ✨ Core Features

- 🤖 **Natural Language Interaction** - Ask in plain English, no commands to memorize 🆕
- 🔄 **Context Sharing** - Share conversation history and preferences across agents 🆕
- 🧠 **Smart Completion** - Auto-fill missing parameters, remember your choices 🆕
- 💡 **Show Thinking Process** - AI shows how it understands and extracts keywords 🆕
- 🔍 **Deep Data Collection** - Use OpenDeepSearch for comprehensive web search
- 🎯 **AI-Driven Prediction** - Hybrid predictions combining LLM reasoning and statistical models
- 📊 **Smart Parameter Extraction** - Automatically identify intent and key information
- 💰 **Betting Advice** - Kelly Criterion-based bankroll management and EV calculation (sports domain)
- ⚡ **FastAPI Interface** - RESTful API support, easy integration
- 🎛️ **Multiple Interaction Methods** - Natural language, CLI, Python API
- 🌐 **Multi-Domain Support** - Sports, weather, elections, **any topic** 🆕

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│          FastAPI REST API               │
│        (HTTP/JSON Interface)            │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────┴──────────────────────────┐
│       PredictionAgent (Core)            │
│  - Workflow Orchestration               │
│  - Risk Management                      │
└──────────────┬──────────────────────────┘
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
┌────────┐ ┌──────┐ ┌──────────┐
│Data    │ │Feature│ │Prediction│
│Collector│ │Extract│ │Engine    │
│(OpenDS)│ │      │ │(LLM+Stats)│
└────────┘ └──────┘ └──────────┘
```

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


**Supported Commands:**
- `/help` - Show help
- `/history` - View conversation history
- `/context` - View current context
- `/clear` - Clear context
- `/set <key> <value>` - Set preference
- `/quit` or `/exit` - Exit

