"""Google News RSS search.

No API key is needed. Supports ``after:``/``before:`` date operators, so it
also works for historical (backtest) windows without look-ahead. A-share/HK
tickers are searched by their Chinese name in the zh-CN edition; other
tickers by their English company name in the en-US edition.
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from .news_utils import (
    HTTP_HEADERS,
    clean_cn_name,
    contains_cjk,
    format_articles,
    get_cn_stock_name,
    get_company_name,
    interleave,
    looks_like_ticker,
    make_article,
    parse_cn_symbol,
    parse_date_range,
    prepare_articles,
    short_error,
    spread_by_day,
)
from .vendor_errors import VendorNoDataError, VendorUnavailableError

RSS_URL = "https://news.google.com/rss/search"
LOCALES = {
    "zh": {"hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"},
    "en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
}
REQUEST_TIMEOUT = 15
MAX_ARTICLES = 30

GLOBAL_NEWS_QUERIES = [
    ("en", "stock market economy"),
    ("en", '"Federal Reserve" interest rates'),
    ("en", "inflation economic outlook"),
    ("zh", "港股 恒生指数"),
    ("zh", "A股 宏观 政策"),
]


def _parse_item(item: ET.Element) -> dict | None:
    try:
        pub_date = parsedate_to_datetime(item.findtext("pubDate", ""))
    except (TypeError, ValueError):
        return None
    if pub_date.tzinfo is not None:
        pub_date = pub_date.astimezone(timezone.utc).replace(tzinfo=None)

    publisher = (item.findtext("source") or "").strip()
    title = (item.findtext("title") or "").strip()
    # Google appends " - Publisher" to every headline.
    head, sep, _ = title.rpartition(" - ")
    if sep and head:
        title = head

    return make_article(
        title=title,
        publisher=publisher or "Google News",
        # Links are long opaque news.google.com redirects; omitted to save tokens.
        pub_date=pub_date,
    )


def _search(session: requests.Session, query: str, start_dt: datetime, end_exclusive: datetime, locale: str) -> list[dict]:
    # Pad the operators by a day each side; results are filtered precisely afterwards.
    dated_query = (
        f"{query} after:{(start_dt - timedelta(days=1)):%Y-%m-%d} "
        f"before:{(end_exclusive + timedelta(days=1)):%Y-%m-%d}"
    )
    try:
        resp = session.get(
            RSS_URL,
            params={"q": dated_query, **LOCALES[locale]},
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        items = ET.fromstring(resp.content).findall("./channel/item")
    except (requests.RequestException, ET.ParseError) as exc:
        raise VendorUnavailableError(f"Google News search failed for {query!r}: {short_error(exc)}") from exc
    return [a for a in map(_parse_item, items) if a]


def _ticker_query(ticker: str) -> tuple[str, str]:
    """Return ``(query, locale)`` for a ticker or a free-text query passed by the agent."""
    symbol = parse_cn_symbol(ticker)
    if symbol:
        name = get_cn_stock_name(symbol)
        if name:
            return f'"{clean_cn_name(name)}"', "zh"
    if looks_like_ticker(ticker):
        name = get_company_name(ticker)
        return (f'"{name}"' if name else f"{ticker} stock"), "en"
    return ticker, ("zh" if contains_cjk(ticker) else "en")


def get_news_google(ticker: str, start_date: str, end_date: str) -> str:
    """
    Retrieve company news from Google News within a date window.

    Args:
        ticker: Ticker symbol (e.g. "TSM", "1810.HK", "600519.SS")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    start_dt, end_exclusive = parse_date_range(start_date, end_date)
    query, locale = _ticker_query(ticker)

    with requests.Session() as session:
        found = _search(session, query, start_dt, end_exclusive, locale)

    articles = prepare_articles(found, start_dt, end_exclusive, MAX_ARTICLES)
    if not articles:
        raise VendorNoDataError(
            f"no Google News articles for {query} between {start_date} and {end_date}"
        )
    return format_articles(
        f"{ticker} News, from {start_date} to {end_date} (source: Google News, query: {query})", articles
    )


def get_global_news_google(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    """
    Retrieve global/macro news (US and China/HK markets) from Google News.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        Formatted string containing global news articles
    """
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    end_exclusive = curr_dt + timedelta(days=1)
    start_date = start_dt.strftime("%Y-%m-%d")

    per_query: list[list[dict]] = []
    errors: list[str] = []
    with requests.Session() as session:
        for locale, query in GLOBAL_NEWS_QUERIES:
            try:
                found = _search(session, query, start_dt, end_exclusive, locale)
            except VendorUnavailableError as exc:
                errors.append(str(exc))
                continue
            per_query.append(prepare_articles(found, start_dt, end_exclusive, limit * 2))

    # Interleave first so every query is represented, then spread across the window's days.
    pool = interleave(per_query, limit * len(GLOBAL_NEWS_QUERIES))
    articles = spread_by_day(pool, limit)
    if articles:
        return format_articles(
            f"Global Market News, from {start_date} to {curr_date} (source: Google News)", articles
        )
    if errors and not per_query:
        raise VendorUnavailableError("; ".join(errors))
    raise VendorNoDataError(f"no Google News global articles between {start_date} and {curr_date}")
