import logging
from typing import Annotated

import requests

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .eastmoney_news import get_news_eastmoney
from .google_news import get_news_google, get_global_news_google
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .vendor_errors import VendorError

# Configuration and routing logic
from .config import get_config

logger = logging.getLogger(__name__)

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "eastmoney",
    "google_news",
]

# Mapping of methods to their vendor-specific implementations.
# Dict order is also the order in which vendors not named in the config are
# appended to the fallback chain.
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "yfinance": get_news_yfinance,
        "eastmoney": get_news_eastmoney,      # A-share / HK only
        "google_news": get_news_google,
        "alpha_vantage": get_alpha_vantage_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "google_news": get_global_news_google,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

# Methods where "every vendor came up empty" is an expected outcome. They return
# a readable note to the agent instead of raising, because an exception inside a
# tool call aborts the whole graph run.
SOFT_FAIL_METHODS = {"get_news", "get_global_news"}

# For news methods, a vendor result with fewer articles than this is kept but
# the router also queries the next vendor and returns the combined text.
MIN_NEWS_ARTICLES = 5

# Errors that move the router on to the next vendor. VendorError covers empty
# results, unsupported inputs, missing API keys and Alpha Vantage rate limits.
FALLBACK_ERRORS = (VendorError, requests.RequestException)

def _short(exc: Exception, max_len: int = 200) -> str:
    text = str(exc) or type(exc).__name__
    return text if len(text) <= max_len else text[:max_len] + "…"

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    attempts = []
    partial_results = []  # sparse news results, merged with whatever comes next
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            result = impl_func(*args, **kwargs)
        except FALLBACK_ERRORS as exc:
            attempts.append(f"{vendor}: {_short(exc)}")
            logger.info("Vendor '%s' could not serve %s, trying next: %s", vendor, method, exc)
            continue

        if method not in SOFT_FAIL_METHODS:
            return result

        count = getattr(result, "article_count", None)
        if count is not None and count < MIN_NEWS_ARTICLES:
            partial_results.append(str(result))
            attempts.append(f"{vendor}: only {count} article(s)")
            logger.info("Vendor '%s' returned only %d article(s) for %s, also trying next", vendor, count, method)
            continue
        return "\n\n".join(partial_results + [str(result)])

    if partial_results:
        return "\n\n".join(partial_results)

    summary = "; ".join(attempts) or "no vendor implementation configured"
    if method in SOFT_FAIL_METHODS:
        arg_text = ", ".join(str(a) for a in args)
        return f"No data found for {method}({arg_text}) from any data vendor. Attempts: {summary}"
    raise RuntimeError(f"No available vendor for '{method}': {summary}")