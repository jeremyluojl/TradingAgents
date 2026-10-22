"""yfinance-based news data fetching functions."""

from datetime import datetime, timezone
from typing import Iterator

import yfinance as yf
from dateutil.relativedelta import relativedelta

from .news_utils import (
    filter_articles,
    format_articles,
    get_company_name,
    looks_like_ticker,
    make_article,
    parse_date_range,
    prepare_articles,
    short_error,
)
from .stockstats_utils import yf_retry
from .vendor_errors import VendorNoDataError, VendorUnavailableError

TICKER_NEWS_COUNT = 20

GLOBAL_NEWS_QUERIES = [
    "stock market economy",
    "Federal Reserve interest rates",
    "inflation economic outlook",
    "global markets trading",
]


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _extract_article_data(article: dict) -> dict:
    """Normalize a yfinance article.

    ``Ticker.get_news`` returns a nested ``content`` structure; ``yf.Search``
    returns a flat structure with ``providerPublishTime`` (epoch seconds, UTC).
    """
    if "content" in article:
        content = article["content"]
        provider = content.get("provider") or {}
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}

        pub_date = None
        pub_date_str = content.get("pubDate", "")
        if pub_date_str:
            try:
                pub_date = _to_naive_utc(datetime.fromisoformat(pub_date_str.replace("Z", "+00:00")))
            except (ValueError, AttributeError):
                pass

        return make_article(
            title=content.get("title", ""),
            summary=content.get("summary", ""),
            publisher=provider.get("displayName", ""),
            link=url_obj.get("url", ""),
            pub_date=pub_date,
        )

    pub_date = None
    timestamp = article.get("providerPublishTime")
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        pub_date = datetime.fromtimestamp(timestamp, timezone.utc).replace(tzinfo=None)

    return make_article(
        title=article.get("title", ""),
        summary=article.get("summary", ""),
        publisher=article.get("publisher", ""),
        link=article.get("link", ""),
        pub_date=pub_date,
    )


def _is_related(article: dict, ticker: str) -> bool:
    """Search hits carry ``relatedTickers``; keep those tagged with our ticker (or untagged)."""
    related = article.get("relatedTickers")
    if not related:
        return True
    return ticker.strip().upper() in {str(sym).upper() for sym in related}


def _search_queries(ticker: str) -> Iterator[str]:
    """Search by ticker first (works for US listings), then by company name
    (needed for HK/other listings, e.g. ``1810.HK`` -> ``Xiaomi``)."""
    yield ticker
    if looks_like_ticker(ticker):
        name = get_company_name(ticker)
        if name and name.lower() != ticker.lower():
            yield name


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Tries ``Ticker.get_news`` first, then falls back to Yahoo's search endpoint
    (by ticker, then by company name). Yahoo only serves recent articles, so
    historical windows usually come back empty and the router moves on.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles

    Raises:
        VendorNoDataError / VendorUnavailableError so the router can fall back.
    """
    start_dt, end_exclusive = parse_date_range(start_date, end_date)
    collected: list[dict] = []
    errors: list[str] = []

    try:
        feed = yf_retry(lambda: yf.Ticker(ticker).get_news(count=TICKER_NEWS_COUNT)) or []
        collected.extend(_extract_article_data(a) for a in feed)
    except Exception as exc:  # yfinance surfaces many network/parse error types
        errors.append(f"Ticker.get_news: {short_error(exc)}")

    if not filter_articles(collected, start_dt, end_exclusive):
        check_related = looks_like_ticker(ticker)
        for query in _search_queries(ticker):
            try:
                search = yf_retry(lambda q=query: yf.Search(
                    query=q,
                    news_count=TICKER_NEWS_COUNT,
                    enable_fuzzy_query=False,
                ))
                hits = search.news or []
            except Exception as exc:
                errors.append(f"Search({query!r}): {short_error(exc)}")
                continue
            collected.extend(
                _extract_article_data(a) for a in hits
                if not check_related or _is_related(a, ticker)
            )
            if filter_articles(collected, start_dt, end_exclusive):
                break

    articles = prepare_articles(collected, start_dt, end_exclusive, TICKER_NEWS_COUNT)
    if articles:
        return format_articles(
            f"{ticker} News, from {start_date} to {end_date} (source: Yahoo Finance)", articles
        )
    if errors and not collected:
        raise VendorUnavailableError("; ".join(errors))
    raise VendorNoDataError(f"no Yahoo Finance articles for {ticker} between {start_date} and {end_date}")


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Yahoo search only returns recent articles; for historical dates nothing
    falls inside the window and the router moves on to the next vendor.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        Formatted string containing global news articles
    """
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    end_exclusive = curr_dt + relativedelta(days=1)
    start_date = start_dt.strftime("%Y-%m-%d")

    collected: list[dict] = []
    errors: list[str] = []
    for query in GLOBAL_NEWS_QUERIES:
        try:
            search = yf_retry(lambda q=query: yf.Search(
                query=q,
                news_count=limit,
                enable_fuzzy_query=True,
            ))
            hits = search.news or []
        except Exception as exc:
            errors.append(f"Search({query!r}): {short_error(exc)}")
            continue
        collected.extend(_extract_article_data(a) for a in hits)
        if len(prepare_articles(collected, start_dt, end_exclusive, limit)) >= limit:
            break

    articles = prepare_articles(collected, start_dt, end_exclusive, limit)
    if articles:
        return format_articles(
            f"Global Market News, from {start_date} to {curr_date} (source: Yahoo Finance)", articles
        )
    if errors and not collected:
        raise VendorUnavailableError("; ".join(errors))
    raise VendorNoDataError(f"no Yahoo Finance global news between {start_date} and {curr_date}")
