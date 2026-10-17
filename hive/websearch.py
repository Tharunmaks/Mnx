"""Web search for answers that need today's facts: news, weather, prices, scores, releases.

DuckDuckGo's plain HTML results (no key, no account), then the top pages themselves, read as text. Pages on private
or local addresses are never fetched (the same rule as the Bees' browsers).
"""

from __future__ import annotations

import asyncio
import html
import re
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

SEARCH_URL = "https://html.duckduckgo.com/html/?q={q}"
UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36"
MAX_PAGE = 1_500_000
transport: httpx.AsyncBaseTransport | None = None  # tests swap in a fake web

RESULT = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>(?P<rest>.*?)(?=<a[^>]+class="result__a"|$)', re.S)
SNIPPET = re.compile(r'class="result__snippet"[^>]*>(?P<s>.*?)</a>', re.S)
DROP = re.compile(r"<(script|style|noscript|svg|nav|footer|header|form|iframe)\b.*?</\1>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")


class SearchError(RuntimeError):
    pass


def _text(fragment: str) -> str:
    return " ".join(html.unescape(TAG.sub(" ", fragment or "")).split())


def _real_url(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    u = urlparse(href)
    if u.netloc.endswith("duckduckgo.com") and u.path.startswith("/l/"):
        return parse_qs(u.query).get("uddg", [href])[0]
    return href


def parse_results(page: str, limit: int = 6) -> list[dict]:
    out = []
    for m in RESULT.finditer(page or ""):
        url = _real_url(m.group("href"))
        if not url.startswith(("http://", "https://")) or "duckduckgo.com/y.js" in url:  # ads
            continue
        sm = SNIPPET.search(m.group("rest"))
        out.append({"title": _text(m.group("title"))[:150], "url": url, "snippet": _text(sm.group("s"))[:400] if sm else ""})
        if len(out) >= limit:
            break
    return out


async def search(query: str, limit: int = 6) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True, transport=transport, headers={"User-Agent": UA}) as c:
            r = await c.get(SEARCH_URL.format(q=quote_plus(query[:300])))
    except httpx.HTTPError as exc:
        raise SearchError(f"The search didn't answer: {exc}") from exc
    if r.status_code >= 400:
        raise SearchError(f"The search answered {r.status_code}")
    return parse_results(r.text, limit)


def page_text(raw: str, limit: int = 3000) -> str:
    body = DROP.sub(" ", raw or "")
    m = re.search(r"<(main|article)\b.*?</\1>", body, re.S | re.I)  # the main content when the page marks it
    return _text(m.group(0) if m and len(_text(m.group(0))) > 300 else body)[:limit]


async def read(url: str, limit: int = 3000) -> str:
    from .browser import _blocked_host
    host = urlparse(url).hostname or ""
    if await asyncio.to_thread(_blocked_host, host):
        return ""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True, transport=transport, headers={"User-Agent": UA}) as c:
            async with c.stream("GET", url) as r:
                if r.status_code >= 400 or not re.search(r"text/(html|plain)", r.headers.get("content-type", "")):
                    return ""
                buf = b""
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > MAX_PAGE:
                        break
    except httpx.HTTPError:
        return ""
    return page_text(buf.decode(errors="replace"), limit)


async def research(query: str, pages: int = 3) -> list[dict]:
    """Results with the text of the top pages: [{n, title, url, snippet, text}]."""
    results = await search(query)
    texts = await asyncio.gather(*(read(r["url"]) for r in results[:pages]))
    return [{"n": i + 1, **r, "text": texts[i] if i < len(texts) else ""} for i, r in enumerate(results)]
