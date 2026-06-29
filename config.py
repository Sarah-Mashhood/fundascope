import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Key configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Model configuration for LLM nodes
# Easily switchable between "gemini-3.1-flash-lite" and "gemini-3.5-flash"
MODEL_NAME = "gemini-3.1-flash-lite"
THINKING_LEVEL = "medium"

# Fixed metric list for fundamental stock analysis
METRIC_FIELDS = [
    "company_name",
    "sector",
    "market_cap",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "revenue_growth_yoy_quarterly",
    "revenue_growth_yoy_annual",
    "eps_growth_yoy_quarterly",
    "eps_growth_yoy_annual",
    "debt_to_equity",
    "current_ratio",
    "total_cash",
    "pe_ratio",
    "ps_ratio",
    "pb_ratio",
]

# Common company name to ticker mappings for resolution
TICKER_MAP = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
}
