"""
Configuration management for the Customer Churn Chatbot.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

# Database Configuration
_db_path = os.getenv("DATABASE_PATH", "data/customer_churn.db")
DATABASE_PATH = str(PROJECT_ROOT / _db_path) if not os.path.isabs(_db_path) else _db_path

# Model Paths
_model_path = os.getenv("MODEL_PATH", "src/models/best_xgboost_model.pkl")
MODEL_PATH = str(PROJECT_ROOT / _model_path) if not os.path.isabs(_model_path) else _model_path

_pipeline_path = os.getenv("PIPELINE_PATH", "src/models/preprocessor_pipeline.pkl")
PIPELINE_PATH = str(PROJECT_ROOT / _pipeline_path) if not os.path.isabs(_pipeline_path) else _pipeline_path

# Streamlit Configuration
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
STREAMLIT_SERVER_HEADLESS = os.getenv("STREAMLIT_SERVER_HEADLESS", "true").lower() == "true"

# Agent Configuration
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", 0.7))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 15))  # Increased for complex queries
AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

# Validation
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    print("⚠️  Warning: OPENAI_API_KEY not set in .env file!")
    print("   Please add your OpenAI API key to the .env file")

def validate_paths():
    """Validate that required files exist."""
    issues = []
    
    if not Path(DATABASE_PATH).exists():
        issues.append(f"Database not found: {DATABASE_PATH}")
    
    if not Path(MODEL_PATH).exists():
        issues.append(f"Model not found: {MODEL_PATH}")
    
    if not Path(PIPELINE_PATH).exists():
        issues.append(f"Pipeline not found: {PIPELINE_PATH}")
    
    return issues

def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"OpenAI Model: {OPENAI_MODEL}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Pipeline: {PIPELINE_PATH}")
    print(f"Agent Temperature: {AGENT_TEMPERATURE}")
    print(f"Agent Max Iterations: {AGENT_MAX_ITERATIONS}")
    print("=" * 60)
