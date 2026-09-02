# Real-Time Financial Fraud & AML Risk Engine

An end-to-end MLOps service that evaluates financial transactions for fraud risk using XGBoost, generates explainable feature attributions with TreeSHAP, and synthesizes audit-ready Anti-Money Laundering (AML) compliance reports using Groq LLM endpoints.

## Architecture

1. **REST API Gateway**: Built with FastAPI and Pydantic for strict request payload validation.
2. **Explainable AI (XAI)**: Evaluates XGBoost risk models and uses TreeSHAP to isolate top feature drivers behind high-risk transaction flags.
3. **AI Compliance Agent**: Utilizes Groq LLM endpoints (via OpenAI client) to auto-generate structured, 3-sentence compliance summaries for financial intelligence teams.

## Tech Stack

* **Language**: Python 3.11+
* **API Framework**: FastAPI, Uvicorn
* **Machine Learning & XAI**: XGBoost, SHAP, Pandas, Scikit-learn
* **LLM Integration**: Groq API, OpenAI Python Client
* **Environment Management**: Python Dotenv, Virtual Environment (`venv`)

## Getting Started

### 1. Prerequisites
Ensure Python 3.11+ and Git are installed.

### 2. Installation
```bash
# Clone repository
git clone [https://github.com/raulshaxel/fraud-risk-engine.git](https://github.com/raulshaxel/fraud-risk-engine.git)
cd fraud-risk-engine

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt