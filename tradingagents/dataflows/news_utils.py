"""Shared helpers for news vendors.

- Article normalization, date filtering, de-duplication and formatting.
- Ticker / company-name resolution, including A-share and Hong Kong symbols
  (e.g. ``1810.HK`` -> 小米集团, ``600519.SS`` -> 贵州茅台).
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import requests
import yfinance as yf

from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
}
SUMMARY_MAX_CHARS = 300


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def make_article(
    title: str,
    summary: str = "",
    publisher: str = "Unknown",
    link: str = "",
    pub_date: Optional[datetime] = None,
) -> dict:
    """Build the normalized article dict shared by all news vendors.

    ``pub_date`` must be a naive datetime (vendors convert to UTC or local
    exchange time before calling this).
    """
    return {
        "title": (title or "").strip() or "No title",
        "summary": (summary or "").strip(),
        "publisher": (publisher or "").strip() or "Unknown",
        "link": (link or "").strip(),
        "pub_date": pub_date,
    }


def parse_date_range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Parse yyyy-mm-dd bounds into ``[start, end_exclusive)``; the end day is inclusive."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_exclusive = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    return start_dt, end_exclusive


def filter_articles(articles: Iterable[dict], start_dt: datetime, end_exclusive: datetime) -> list[dict]:
    """Keep dated articles inside the window. Undated ones are dropped to avoid look-ahead."""
    return [
        a for a in articles
        if a.get("pub_date") is not None and start_dt <= a["pub_date"] < end_exclusive
    ]


def _title_key(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


def dedupe_articles(articles: Iterable[dict]) -> list[dict]:
    """Drop articles whose normalized title was already seen (first occurrence wins)."""
    seen, unique = set(), []
    for article in articles:
        key = _title_key(article["title"])
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


def spread_by_day(articles: list[dict], limit: int) -> list[dict]:
    """Pick up to ``limit`` articles round-robin across days, so one busy day
    does not crowd out the rest of the window. Result is sorted newest first."""
    by_day: dict = {}
    for article in sorted(articles, key=lambda a: a["pub_date"], reverse=True):
        by_day.setdefault(article["pub_date"].date(), []).append(article)

    buckets = [by_day[day] for day in sorted(by_day, reverse=True)]
    picked: list[dict] = []
    while len(picked) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(picked) < limit:
                picked.append(bucket.pop(0))
    return sorted(picked, key=lambda a: a["pub_date"], reverse=True)


def interleave(article_lists: list[list[dict]], limit: int) -> list[dict]:
    """Round-robin merge of several (already ranked) lists, de-duplicated by title."""
    merged, seen = [], set()
    queues = [list(lst) for lst in article_lists]
    while len(merged) < limit and any(queues):
        for queue in queues:
            while queue:
                article = queue.pop(0)
                key = _title_key(article["title"])
                if key not in seen:
                    seen.add(key)
                    merged.append(article)
                    break
            if len(merged) >= limit:
                break
    return merged


def prepare_articles(articles: Iterable[dict], start_dt: datetime, end_exclusive: datetime, limit: int) -> list[dict]:
    """Date-filter, de-duplicate and cap a vendor's raw article list."""
    return spread_by_day(dedupe_articles(filter_articles(articles, start_dt, end_exclusive)), limit)


class NewsResult(str):
    """Formatted news text that also records how many articles it contains,
    so the router can keep looking when a vendor only found a handful."""

    article_count: int

    def __new__(cls, text: str, article_count: int):
        obj = super().__new__(cls, text)
        obj.article_count = article_count
        return obj


def short_error(exc: Exception, max_len: int = 160) -> str:
    """Compact error text for fallback summaries (requests errors embed full URLs)."""
    if isinstance(exc, requests.RequestException):
        return type(exc).__name__
    text = str(exc) or type(exc).__name__
    return text if len(text) <= max_len else text[:max_len] + "…"


def format_articles(heading: str, articles: list[dict]) -> NewsResult:
    """Render articles as the Markdown block handed to the agents."""
    lines = [f"## {heading}:", ""]
    for article in articles:
        when = article["pub_date"].strftime("%Y-%m-%d %H:%M") if article.get("pub_date") else "date unknown"
        lines.append(f"### {article['title']} (source: {article['publisher']}, {when})")
        summary = article.get("summary", "")
        if summary:
            if len(summary) > SUMMARY_MAX_CHARS:
                summary = summary[:SUMMARY_MAX_CHARS].rstrip() + "…"
            lines.append(summary)
        if article.get("link"):
            lines.append(f"Link: {article['link']}")
        lines.append("")
    return NewsResult("\n".join(lines), len(articles))


# ---------------------------------------------------------------------------
# Tickers and company names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CNSymbol:
    """An A-share or Hong Kong listing."""

    market: str  # "hk", "sh", "sz" or "bj"
    code: str    # 5 digits for HK, 6 digits for A-shares

    @property
    def tencent_id(self) -> str:
        return f"{self.market}{self.code}"


_A_SHARE_SUFFIXES = {"SS": "sh", "SH": "sh", "SZ": "sz", "BJ": "bj"}


def parse_cn_symbol(ticker: str) -> Optional[CNSymbol]:
    """Recognize ``1810.HK``, ``600519.SS``/``.SH``, ``000001.SZ``, ``430047.BJ`` or a bare 6-digit A-share code."""
    t = (ticker or "").strip().upper()

    match = re.fullmatch(r"(\d{1,5})\.HK", t)
    if match:
        return CNSymbol("hk", match.group(1).zfill(5))

    match = re.fullmatch(r"(\d{6})\.(SS|SH|SZ|BJ)", t)
    if match:
        return CNSymbol(_A_SHARE_SUFFIXES[match.group(2)], match.group(1))

    if re.fullmatch(r"\d{6}", t):
        if t.startswith(("92", "4", "8")):
            return CNSymbol("bj", t)
        if t.startswith(("6", "9")):
            return CNSymbol("sh", t)
        if t.startswith(("0", "2", "3")):
            return CNSymbol("sz", t)
    return None


def looks_like_ticker(text: str) -> bool:
    """True for exchange tickers such as ``AAPL``, ``1810.HK``, ``^GSPC``; False for free-text queries."""
    return bool(re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.\-=]{0,14}", (text or "").strip()))


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


_cn_name_cache: dict[str, str] = {}


def get_cn_stock_name(symbol: CNSymbol) -> Optional[str]:
    """Chinese short name via Tencent's quote API (``hk01810`` -> ``小米集团-W``). Returns None on failure."""
    if symbol.tencent_id in _cn_name_cache:
        return _cn_name_cache[symbol.tencent_id]
    try:
        resp = requests.get(
            f"https://qt.gtimg.cn/q={symbol.tencent_id}", headers=HTTP_HEADERS, timeout=8
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Tencent name lookup failed for %s: %s", symbol.tencent_id, exc)
        return None

    # Response format: v_hk01810="100~小米集团-W~01810~...";
    match = re.search(r'="([^"]*)"', resp.content.decode("gbk", errors="ignore"))
    fields = match.group(1).split("~") if match else []
    name = fields[1].strip() if len(fields) > 2 else ""
    if not name:
        return None
    _cn_name_cache[symbol.tencent_id] = name
    return name


def clean_cn_name(name: str) -> str:
    """Strip listing markers: ``小米集团-W`` -> ``小米集团``, ``*ST康美`` -> ``康美``, ``XD贵州茅`` -> ``贵州茅``."""
    name = unicodedata.normalize("NFKC", name).strip()
    name = re.sub(r"^\*?ST", "", name)
    name = re.sub(r"^(XD|XR|DR|N|C)(?=[\u4e00-\u9fff])", "", name)
    name = re.sub(r"-[A-Z]{1,3}$", "", name)
    return name.strip()


def core_cn_name(name: str) -> str:
    """Drop generic corporate suffixes for relevance matching: ``小米集团`` -> ``小米``."""
    core = re.sub(r"(集团|控股|股份)$", "", name)
    return core if len(core) >= 2 else name


_EN_SUFFIX = re.compile(
    r"[,\s]+(inc\.?|incorporated|corporation|corp\.?|company|co\.?|ltd\.?|limited|plc|"
    r"holdings?|group|n\.v\.|s\.a\.|se|ag|sa)$",
    re.IGNORECASE,
)


def clean_en_name(name: str) -> str:
    """``Taiwan Semiconductor Manufacturing Company Limited`` -> ``Taiwan Semiconductor Manufacturing``."""
    name = name.strip()
    while True:
        stripped = _EN_SUFFIX.sub("", name).strip(" ,")
        if not stripped or stripped == name:
            return name
        name = stripped


_en_name_cache: dict[str, str] = {}


def get_company_name(ticker: str) -> Optional[str]:
    """Cleaned English company name from Yahoo Finance (``1810.HK`` -> ``Xiaomi``). Returns None on failure."""
    key = ticker.strip().upper()
    if key in _en_name_cache:
        return _en_name_cache[key]
    try:
        info = yf_retry(lambda: yf.Ticker(key).info) or {}
    except Exception as exc:  # yfinance raises a variety of network/parse errors
        logger.info("Yahoo company-name lookup failed for %s: %s", key, exc)
        return None
    raw = info.get("longName") or info.get("shortName")
    if not raw:
        return None
    name = clean_en_name(raw)
    _en_name_cache[key] = name
    return name
