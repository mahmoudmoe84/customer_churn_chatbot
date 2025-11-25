# Customer Churn Prediction Chatbot 🤖

An intelligent, agent-based chatbot that uses LLM reasoning to analyze customer churn, query databases, and provide predictive insights through natural language interactions.

## 🎯 Features

- **Natural Language Queries**: Ask questions in plain English
- **Intelligent Agent**: LLM-powered reasoning with LangChain
- **Real-time Predictions**: XGBoost model with optimized hyperparameters
- **Database Integration**: Queries fresh customer data from SQLite
- **Interactive UI**: Streamlit-based chat interface
- **Reasoning Transparency**: See how the agent makes decisions

## 🏗️ Project Structure

```
customer_churn_chatbot/
├── data/
│   ├── Telco_Customer_Churn.csv      # Raw data
│   └── customer_churn.db             # SQLite database
├── notebooks/
│   └── 01_eda.ipynb                  # EDA and model training
├── src/
│   ├── agents/                       # LLM agent logic
│   ├── database/                     # Database queries
│   ├── models/                       # Model & pipeline files
│   ├── tools/                        # Agent tools (functions)
│   ├── prompts/                      # System prompts
│   ├── config.py                     # Configuration management
│   └── app.py                        # Streamlit interface
├── .env                              # Environment variables (API keys)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd customer_churn_chatbot
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the App

```bash
streamlit run src/app.py
```

## 🔧 Configuration

Edit `.env` file to customize:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model version (default: gpt-4-turbo-preview)
- `AGENT_TEMPERATURE`: Response creativity (0.0-1.0)
- `AGENT_MAX_ITERATIONS`: Max reasoning steps

## 💬 Example Queries

- "What is the current churn rate?"
- "Predict churn for customer ID 1234"
- "Show me high-risk customers"
- "What factors influence churn the most?"
- "Calculate churn rate for senior citizens"

## 🧠 How It Works

1. **User asks a question** in natural language
2. **LLM Agent** analyzes the query and decides which tools to use
3. **Tools execute**:
   - Query database for customer data
   - Run ML model predictions
   - Calculate metrics and statistics
4. **Agent reasons** about the results
5. **Response** is generated with insights and visualizations

## 📊 Model Details

- **Algorithm**: XGBoost Classifier
- **Optimization**: Optuna (100 trials)
- **Features**: 29 (after preprocessing)
- **Metrics**: AUC, Accuracy, Precision, Recall, F1

## 🛠️ Development

### Run Tests
```bash
pytest tests/
```

### Train New Model
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## 📝 License

MIT License

## 👤 Author

Mahmoud Mohamed
