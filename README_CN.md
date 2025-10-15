# 🎯 预测AI智能体

[English](README.md) | [中文](README_CN.md)

> 🌐 **通用预测系统** - 支持体育、天气、选举等多领域预测

基于 [OpenDeepSearch](https://github.com/sentient-agi/OpenDeepSearch) 的智能多领域预测系统

## 🚀 快速开始

```bash
# 一键启动聊天
python3 chat.py

# 然后开始对话
💬 您: 预测明天北京的天气
💬 您: 巴塞罗那对皇马谁会赢
💬 您: 后天呢？  # 自动记住上下文
```

## ✨ 核心功能

- 🤖 **自然语言交互** - 用自然语言提问，无需记忆命令 🆕
- 🔄 **上下文共享** - 在不同智能体间共享对话历史和偏好 🆕
- 🧠 **智能补全** - 自动填充缺失参数，记住您的选择 🆕
- 💡 **展示思考过程** - AI显示如何理解和提取关键词 🆕
- 🔍 **深度数据采集** - 使用 OpenDeepSearch 进行全面网络搜索
- 🎯 **AI驱动预测** - 结合 LLM 推理和统计模型的混合预测
- 📊 **智能参数提取** - 自动识别意图和关键信息
- 💰 **投注建议** - 基于凯利准则的资金管理和期望价值计算（体育领域）
- ⚡ **FastAPI接口** - RESTful API支持，易于集成
- 🎛️ **多种交互方式** - 自然语言、命令行、Python API
- 🌐 **多领域支持** - 体育、天气、选举、**任意主题** 🆕

## 🏗️ 系统架构

```
┌─────────────────────────────────────────┐
│          FastAPI REST API               │
│        (HTTP/JSON 接口)                 │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────┴──────────────────────────┐
│       PredictionAgent (核心)            │
│  - 工作流编排                            │
│  - 风险管理                              │
└──────────────┬──────────────────────────┘
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
┌────────┐ ┌──────┐ ┌──────────┐
│数据    │ │特征  │ │预测      │
│采集器  │ │提取  │ │引擎      │
│(OpenDS)│ │      │ │(LLM+统计)│
└────────┘ └──────┘ └──────────┘
```

## 🚀 开始使用

### 1. 安装依赖

```bash
cd prediction-ai-agent
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `env.example.txt` 创建 `.env` 文件：

```bash
cp env.example.txt .env
```

编辑 `.env` 文件并填入您的 API 密钥：

```env
# 必需的 API 密钥
SERPER_API_KEY=your-serper-api-key
JINA_API_KEY=your-jina-api-key
OPENROUTER_API_KEY=your-openrouter-api-key

# LLM 模型配置（使用 Gemini 2.0 Flash）
LITELLM_MODEL_ID=openrouter/google/gemini-2.0-flash-001
OPENROUTER_MODEL=google/gemini-2.0-flash-001

# 搜索配置
SEARCH_PROVIDER=serper
RERANKER=jina
```

### 3. 使用方法

#### 方法 A：聊天模式（推荐）🆕

**最简单的方式 - 像聊天一样提问！**

```bash
# 启动聊天
python3 chat.py

# 然后开始对话：
💬 您: 预测明天北京的天气
💡 AI理解: 查询北京的天气
   关键词: {"location":"北京","date":"2025-10-16"}
   置信度: 95%
🌤️  北京天气: 晴天
   温度: 15-22°C
   降水概率: 20%

💬 您: 后天呢？
💡 AI理解: 查询北京的天气  # 自动记住"北京"
🌤️  北京天气: 多云
   温度: 16-23°C

💬 您: 比特币会涨吗
💡 AI理解: 通用预测：比特币会涨吗
   关键词: {"query":"比特币会涨吗","topic":"比特币"}
🔮 通用预测
   结果: 短期内可能上涨...
   概率: 65%

💬 您: /help             # 显示帮助
💬 您: /quit             # 退出
```

**支持的命令：**
- `/help` - 显示帮助
- `/history` - 查看对话历史
- `/context` - 查看当前上下文
- `/clear` - 清除上下文
- `/set <键> <值>` - 设置偏好
- `/quit` 或 `/exit` - 退出

#### 方法 B：命令行模式

**天气预测：**
```bash
python3 universal_predict.py weather --location "北京" --date "2025-10-20" --days 7
```

**体育赛事预测：**
```bash
python3 universal_predict.py sports --team1 "巴塞罗那" --team2 "皇家马德里" --league "西甲"
```

**选举预测：**
```bash
python3 universal_predict.py election --election "2024美国大选" --region "美国" --candidates "候选人A" "候选人B"
```

#### 方法 C：Python API

```python
from src.universal_agent import UniversalPredictionAgent

# 初始化智能体
agent = UniversalPredictionAgent()

# 天气预测
result = agent.predict(
    domain="weather",
    params={
        "location": "北京",
        "date": "2025-10-20",
        "days_ahead": 7
    },
    use_search=True
)

# 体育预测
result = agent.predict(
    domain="sports",
    params={
        "team1": "巴塞罗那",
        "team2": "皇家马德里",
        "league": "西甲"
    },
    use_search=True
)

# 通用预测（任何主题）
result = agent.predict(
    domain="general",
    params={
        "query": "比特币会涨吗？",
        "topic": "加密货币"
    },
    use_search=True
)

print(result)
```

#### 方法 D：FastAPI 服务

```bash
# 启动 API 服务
python3 main.py api --host 0.0.0.0 --port 8000

# API 文档地址
# http://localhost:8000/docs
```

## 📊 预测领域

### 1. 天气预测 🌤️
- 支持全球任意城市
- 多日天气预报
- 考虑地理位置和季节性因素

### 2. 体育赛事预测 ⚽
- 足球、篮球等主流赛事
- 考虑历史对战、近期表现、主客场等因素
- 提供投注建议和期望价值分析

### 3. 选举预测 🗳️
- 支持各类选举预测
- 基于民调、专家分析、历史数据
- 识别所有候选人并给出概率分布

### 4. 通用预测 🔮 🆕
- **支持任意主题的预测**
- 加密货币、股票、科技趋势、社会事件等
- AI自动分析并给出预测
- 锦标赛预测、冠军预测等

## 🎯 核心特性

### 自然语言理解 🆕
系统能够理解自然语言输入，自动识别预测领域和提取关键参数：

```
输入: "预测明天纽约的天气"
识别: 领域=天气, 地点=纽约, 日期=明天

输入: "巴萨对皇马谁会赢"
识别: 领域=体育, 队伍1=巴塞罗那, 队伍2=皇家马德里

输入: "谁会赢得2025世界杯冠军"
识别: 领域=通用, 查询=世界杯冠军预测
```

### 上下文管理 🆕
- **对话历史记录**：记住之前的对话
- **智能参数补全**：自动填充缺失的参数
- **用户偏好记忆**：记住用户的设置和偏好
- **跨领域上下文共享**：在不同预测任务间共享信息

### 展示思考过程 🆕
AI会显示它如何理解您的问题：
```
💡 AI理解:
   领域: 天气
   关键词: {"location":"北京","date":"2025-10-16"}
   置信度: 95%
   自动补全: 无
```

## 🔧 高级配置

### 环境变量说明

```env
# OpenRouter配置（使用Gemini 2.0 Flash，支持图像）
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=google/gemini-2.0-flash-001
LITELLM_MODEL_ID=openrouter/google/gemini-2.0-flash-001

# 搜索服务
SERPER_API_KEY=your-serper-key      # Google搜索API
JINA_API_KEY=your-jina-key          # Jina重排序API

# 搜索提供商选择
SEARCH_PROVIDER=serper              # serper 或 searxng
RERANKER=jina                       # jina 或 none

# 可选：OpenDeepSearch配置
ENABLE_OPENDEEPSEARCH=false         # 启用深度搜索（需要额外依赖）
```

### Docker 部署

```bash
# 使用 docker-compose
docker-compose up -d

# 或使用 Docker
docker build -t prediction-agent .
docker run -p 8000:8000 --env-file .env prediction-agent
```

## 📚 项目结构

```
prediction-ai-agent/
├── chat.py                      # 🆕 聊天模式入口
├── main.py                      # 主程序入口
├── universal_predict.py         # 🆕 通用预测CLI
├── smart_predict.py             # 🆕 智能预测（带上下文）
├── requirements.txt             # 依赖列表
├── .env                        # 环境变量配置
├── README.md                   # 英文文档
├── README_CN.md                # 中文文档
├── src/
│   ├── agent.py                # 核心Agent
│   ├── universal_agent.py      # 🆕 通用预测Agent
│   ├── nlp_parser.py           # 🆕 自然语言解析器
│   ├── context_manager.py      # 🆕 上下文管理器
│   ├── data_collector.py       # 数据采集
│   ├── feature_extractor.py    # 特征提取
│   ├── prediction_engine.py    # 预测引擎
│   ├── search_client.py        # 🆕 搜索客户端
│   ├── openai_client.py        # OpenRouter客户端
│   ├── api.py                  # FastAPI接口
│   └── domains/                # 🆕 预测领域
│       ├── sports.py           # 体育预测
│       ├── weather.py          # 天气预测
│       ├── election.py         # 选举预测
│       └── general.py          # 🆕 通用预测
├── examples/                    # 示例代码
├── tests/                       # 测试
└── config/                      # 配置文件
```

## 🛠️ 技术栈

- **LLM**: OpenRouter (Gemini 2.0 Flash - 支持图像、快速响应)
- **搜索**: Serper API (Google搜索)
- **深度搜索**: OpenDeepSearch (可选)
- **重排序**: Jina Reranker
- **框架**: FastAPI, Pydantic, LiteLLM
- **日志**: Loguru

## 📈 使用示例

### 完整工作流示例

```python
from src.universal_agent import UniversalPredictionAgent
from src.context_manager import ContextManager

# 创建智能体和上下文管理器
agent = UniversalPredictionAgent()
ctx = ContextManager(user_id="user123")

# 第一次预测：天气
result1 = agent.predict(
    domain="weather",
    params={"location": "北京", "date": "2025-10-20"},
    use_search=True
)
ctx.add_prediction("weather", {"location": "北京"}, result1)

# 第二次预测：智能补全（记住"北京"）
params = {"date": "2025-10-21"}  # 没有指定地点
params = ctx.smart_complete_params("weather", params)  # 自动补全为"北京"

result2 = agent.predict(
    domain="weather",
    params=params,
    use_search=True
)

# 设置用户偏好
ctx.set_preference("default_location", "上海")
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🔗 相关链接

- [OpenDeepSearch](https://github.com/sentient-agi/OpenDeepSearch)
- [OpenRouter](https://openrouter.ai/)
- [Serper API](https://serper.dev/)
- [Jina AI](https://jina.ai/)

## 💡 提示

1. **获取API密钥**：
   - OpenRouter: https://openrouter.ai/
   - Serper: https://serper.dev/
   - Jina: https://jina.ai/

2. **推荐模型**：Gemini 2.0 Flash（快速、便宜、支持图像）

3. **聊天模式最简单**：直接运行 `python3 chat.py` 开始使用

4. **上下文记忆**：系统会自动记住您的偏好和历史对话

5. **通用预测**：可以预测任何主题，不仅限于体育、天气、选举

## 🆕 最新更新

- ✅ 新增聊天模式（`chat.py`）- 最简单的交互方式
- ✅ 新增自然语言理解 - 无需记忆命令
- ✅ 新增上下文管理 - 智能记忆和补全
- ✅ 新增通用预测领域 - 支持任意主题
- ✅ 展示AI思考过程 - 理解关键词提取
- ✅ 使用Serper API进行真实搜索 - 获取最新数据
- ✅ 完整的英文翻译 - 代码和文档国际化

## 📞 支持

如有问题，请提交 Issue 或查看文档。

---

⭐ 如果这个项目对您有帮助，请给个 Star！

