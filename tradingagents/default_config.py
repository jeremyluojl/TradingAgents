import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Amazon Bedrock configuration
    "bedrock_region": os.getenv("BEDROCK_REGION"),  # None → boto3 auto-detects (uses EC2 instance region on EC2)
    "bedrock_credentials_profile": os.getenv("AWS_PROFILE") or os.getenv("BEDROCK_CREDENTIALS_PROFILE"),
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # A comma-separated list is a fallback chain: when a vendor errors out or
    # returns nothing, the next one is tried; vendors not listed are appended
    # at the end automatically.
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        # Options: eastmoney (A-share/HK only, Chinese), yfinance, google_news, alpha_vantage (needs ALPHA_VANTAGE_API_KEY)
        "news_data": "eastmoney,yfinance,google_news",
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Google News supports date-bounded search (no look-ahead in backtests)
        # and covers both US and China/HK macro news.
        "get_global_news": "google_news,yfinance",
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
