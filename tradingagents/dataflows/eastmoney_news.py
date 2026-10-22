"""Eastmoney (东方财富) news search for A-share and Hong Kong tickers.

Uses the public JSONP endpoint behind https://so.eastmoney.com/news/s. No API
key is needed. Two searches are combined:

- by stock code, newest first (precise, e.g. ``01810`` / ``600519``);
- by Chinese short name, relevance-sorted (newest-first sorting on a name is
  dominated by unrelated hits, so relevance sorting is used instead).
"""

import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

from .news_utils import (
    HTTP_HEADERS,
    clean_cn_name,
    core_cn_name,
    format_articles,
    get_cn_stock_name,
    make_article,
    parse_cn_symbol,
    parse_date_range,
    prepare_articles,
    short_error,
)
from .vendor_errors import VendorNoDataError, VendorUnavailableError

SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
PAGE_SIZE = 50
MAX_PAGES = 2
MAX_ARTICLES = 30
REQUEST_TIMEOUT = (5, 10)  # (connect, read) seconds
REQUEST_ATTEMPTS = 2       # the endpoint occasionally stalls on the first request


def _clean_text(text: str) -> str:
    text = re.sub(r"</?em>", "", text or "")
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def _search_page(session: requests.Session, keyword: str, sort: str, page: int) -> list[dict]:
    inner = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": sort,  # "time" (newest first) or "default" (relevance)
                "pageIndex": page,
                "pageSize": PAGE_SIZE,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    params = {"cb": "jQuery_tradingagents", "param": json.dumps(inner, ensure_ascii=False)}
    headers = {
        **HTTP_HEADERS,
        "Referer": "https://so.eastmoney.com/news/s?keyword=" + urllib.parse.quote(keyword),
    }

    last_error: Exception | None = None
    for _ in range(REQUEST_ATTEMPTS):
        try:
            resp = session.get(SEARCH_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = resp.text
            payload = json.loads(text[text.index("(") + 1 : text.rindex(")")])  # strip JSONP wrapper
            return (payload.get("result") or {}).get("cmsArticleWebOld") or []
        except (requests.RequestException, ValueError) as exc:  # ValueError covers JSON/wrapper parsing
            last_error = exc
    raise VendorUnavailableError(f"Eastmoney search failed for {keyword!r}: {short_error(last_error)}")


def _to_article(raw: dict) -> dict | None:
    try:
        pub_date = datetime.strptime(raw.get("date", ""), "%Y-%m-%d %H:%M:%S")  # Beijing time
    except ValueError:
        return None
    link = raw.get("url") or (
        f"http://finance.eastmoney.com/a/{raw['code']}.html" if raw.get("code") else ""
    )
    return make_article(
        title=_clean_text(raw.get("title", "")),
        summary=_clean_text(raw.get("content", "")),
        publisher=raw.get("mediaName") or "东方财富",
        link=link,
        pub_date=pub_date,
    )


def _collect(keyword: str, sort: str, start_dt: datetime, relevance_terms: list[str] | None) -> tuple[list[dict], str | None]:
    """Run one paged search. Returns (articles, error message or None)."""
    collected: list[dict] = []
    with requests.Session() as session:
        for page in range(1, MAX_PAGES + 1):
            try:
                hits = _search_page(session, keyword, sort, page)
            except VendorUnavailableError as exc:
                return collected, str(exc)

            page_articles = [a for a in map(_to_article, hits) if a]
            collected.extend(
                a for a in page_articles
                if not relevance_terms
                or any(term in a["title"] + a["summary"] for term in relevance_terms)
            )

            if len(hits) < PAGE_SIZE:
                break
            # Newest-first results: stop once the page reaches back past the window.
            if sort == "time" and page_articles and min(a["pub_date"] for a in page_articles) < start_dt:
                break
    return collected, None


def get_news_eastmoney(ticker: str, start_date: str, end_date: str) -> str:
    """
    Retrieve Chinese-language news for an A-share or HK ticker from Eastmoney.

    Args:
        ticker: e.g. "1810.HK", "600519.SS", "000001.SZ"
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles

    Raises:
        VendorUnavailableError for non-Chinese tickers or network failures,
        VendorNoDataError when nothing falls inside the window.
    """
    symbol = parse_cn_symbol(ticker)
    if symbol is None:
        raise VendorUnavailableError(f"Eastmoney only covers A-share/HK tickers, got {ticker!r}")

    start_dt, end_exclusive = parse_date_range(start_date, end_date)
    raw_name = get_cn_stock_name(symbol)
    name = clean_cn_name(raw_name) if raw_name else ""

    # (keyword, sort, relevance terms). Code hits are trusted as-is; name hits
    # must mention the company, its code, or its core name (小米集团 -> 小米).
    searches = [(symbol.code, "time", None)]
    if name:
        terms = [t for t in dict.fromkeys([name, core_cn_name(name), symbol.code]) if t]
        searches.append((name, "default", terms))

    collected: list[dict] = []
    errors: list[str] = []
    # The endpoint sometimes stalls for several seconds, so run searches concurrently.
    with ThreadPoolExecutor(max_workers=len(searches)) as pool:
        futures = [pool.submit(_collect, kw, sort, start_dt, terms) for kw, sort, terms in searches]
        for future in futures:
            articles, error = future.result()
            collected.extend(articles)
            if error:
                errors.append(error)

    articles = prepare_articles(collected, start_dt, end_exclusive, MAX_ARTICLES)
    if articles:
        label = f"{ticker} ({raw_name})" if raw_name else ticker
        return format_articles(
            f"{label} News, from {start_date} to {end_date} (source: Eastmoney 东方财富)", articles
        )
    if errors and not collected:
        raise VendorUnavailableError("; ".join(errors))
    raise VendorNoDataError(f"no Eastmoney articles for {ticker} between {start_date} and {end_date}")
