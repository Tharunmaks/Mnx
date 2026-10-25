"""The browser agent: open pages, search the web, and optionally drive Chrome.

Off unless you turn it on (``--tools web``). Pages are fetched with the
standard library by default. With ``--chrome`` the agent drives a real Chrome
through Playwright instead: it attaches to the Chrome you already have open
(start it with ``--remote-debugging-port=9222``) so your sign-ins work, or
launches one. Either way it only *reads*: it never types into pages, submits
forms or clicks buttons, and page text is shown to you, never obeyed.

Things it understands::

    open https://example.com            read a page: title, summary, links
    what does https://x.org say about Y the page's sentences that best answer Y
    follow link 3                       open a link from the last page
    search the web for python asyncio   top results from DuckDuckGo (no key)
    latest news about ... / weather in Paris / stock price of ...   a web search
"""

from __future__ import annotations

import atexit
import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, List, Optional, Tuple

from mnx.agent.base import Answer, Skill
from mnx.agent.retrieval import BM25Index, Chunk, split_sentences, tokenize

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) MNX1-agent/1.0 (+https://github.com/Tharunmaks/Mnx)"
SEARCH_URL = "https://html.duckduckgo.com/html/?q="
MAX_BYTES = 3_000_000


@dataclass
class Page:
    url: str
    title: str = ""
    paragraphs: List[str] = field(default_factory=list)
    links: List[Tuple[str, str]] = field(default_factory=list)  # (text, absolute url)


class _TextExtractor(HTMLParser):
    """Visible text, grouped into blocks, plus the page's links."""

    _SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
    _BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br", "section", "article", "blockquote",
              "pre", "td", "dd", "dt", "figcaption", "main"}
    _CHROME = {"nav", "footer", "header", "aside", "form", "button", "select"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.skip = 0
        self.chrome = 0
        self.in_title = False
        self.title = ""
        self.blocks: List[str] = []
        self.buf: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self.link: Optional[List] = None

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        if text:
            self.blocks.append(text)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
        elif tag in self._CHROME:
            self.chrome += 1
        if tag == "title":
            self.in_title = True
        if tag in self._BLOCK:
            self._flush()
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self.link = [urllib.parse.urljoin(self.base, href), []]

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self._CHROME:
            self.chrome = max(0, self.chrome - 1)
        if tag == "title":
            self.in_title = False
        if tag in self._BLOCK:
            self._flush()
        if tag == "a" and self.link is not None:
            text = re.sub(r"\s+", " ", "".join(self.link[1])).strip()
            if text and self.link[0].startswith(("http://", "https://")):
                self.links.append((text, self.link[0]))
            self.link = None

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return
        if self.skip:
            return
        if self.link is not None:
            self.link[1].append(data)
        if not self.chrome:
            self.buf.append(data)


def parse_html(text: str, url: str) -> Page:
    p = _TextExtractor(url)
    p.feed(text)
    p.close()
    p._flush()
    seen, paras = set(), []
    for b in p.blocks:
        if b not in seen:
            seen.add(b)
            paras.append(b)
    links, seen_urls = [], set()
    for t, u in p.links:
        if u not in seen_urls:
            seen_urls.add(u)
            links.append((t, u))
    return Page(url, re.sub(r"\s+", " ", p.title).strip(), paras, links)


# ------------------------------------------------------------------ fetchers
def urllib_fetch(url: str, timeout: float = 15.0) -> Tuple[str, str]:
    """Return (final url, html) using only the standard library."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        if ctype and not re.search(r"text/|html|xml|json", ctype):
            raise ValueError(f"that link is a {ctype.split(';')[0]} file, not a web page")
        raw = r.read(MAX_BYTES)
        charset = r.headers.get_content_charset() or "utf-8"
        body = raw.decode(charset, errors="replace")
        if "html" not in ctype and "<html" not in body[:2000].lower():
            body = "<pre>" + html.escape(body) + "</pre>"
        return r.geturl(), body


class ChromeFetcher:
    """Fetch pages through a real Chrome with Playwright (``pip install playwright``).

    ``cdp`` attaches to a Chrome you started with ``--remote-debugging-port``,
    so pages open in your own window with your sign-ins. Without it, a new
    Chrome is launched (``channel="chrome"``; falls back to Playwright's
    Chromium). Pages open in new tabs; your existing tabs are left alone.
    """

    def __init__(self, cdp: Optional[str] = "http://localhost:9222", headless: bool = False):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # pragma: no cover - depends on the machine
            raise RuntimeError("Chrome control needs Playwright: pip install playwright && playwright install chrome") from e
        self._pw = sync_playwright().start()
        self.browser = None
        self.attached = False
        if cdp:
            try:
                self.browser = self._pw.chromium.connect_over_cdp(cdp)
                self.attached = True
            except Exception:
                self.browser = None
        if self.browser is None:
            try:
                self.browser = self._pw.chromium.launch(channel="chrome", headless=headless)
            except Exception:
                self.browser = self._pw.chromium.launch(headless=headless)
        ctxs = self.browser.contexts
        self.context = ctxs[0] if ctxs else self.browser.new_context()
        self.page = None
        atexit.register(self.close)

    def __call__(self, url: str, timeout: float = 20.0) -> Tuple[str, str]:
        if self.page is None or self.page.is_closed():
            self.page = self.context.new_page()
        self.page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
        return self.page.url, self.page.content()

    def close(self):
        """Close MNX's tab; quit the browser only if MNX launched it (never your own Chrome)."""
        if self._pw is None:
            return
        try:
            if self.attached:
                if self.page is not None and not self.page.is_closed():
                    self.page.close()
            else:
                self.browser.close()
        except Exception:
            pass
        finally:
            self._pw.stop()
            self._pw = None


# ------------------------------------------------------------------ search
def parse_search_results(text: str) -> List[Tuple[str, str, str]]:
    """(title, url, snippet) from DuckDuckGo's HTML results page."""
    out = []
    for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="[^"]*result__a|\Z)',
                         text, re.S):
        href, title, rest = m.groups()
        href = html.unescape(href)
        q = urllib.parse.urlparse(href)
        if "uddg" in urllib.parse.parse_qs(q.query):  # DuckDuckGo's redirect wrapper
            href = urllib.parse.parse_qs(q.query)["uddg"][0]
        if href.startswith("//"):
            href = "https:" + href
        if "duckduckgo.com/y.js" in href:  # ads
            continue
        snip = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(a|div|td)>', rest, re.S)
        clean = lambda s: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()
        out.append((clean(title), href, clean(snip.group(1)) if snip else ""))
    return out


# ------------------------------------------------------------------ skill
_URL = r"(https?://\S+|www\.\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|ai|edu|gov|in|co|uk|app)(?:/\S*)?)"
_OPEN = re.compile(rf"^(?:please\s+)?(?:open|read|visit|go to|browse(?: to)?|fetch|load|show me|summari[sz]e)\s+(?:the page\s+)?{_URL}"
                   r"(?:\s+in (?:chrome|the browser))?\W*$", re.I)
_ASK_PAGE = re.compile(rf"^(?:what does|what do(?:es)? the page)\s+{_URL}\s+say about\s+(.+?)\W*$|"
                       rf"^(?:on|in|from)\s+{_URL}\s*[,:]\s*(.+?)\W*$|"
                       rf"^(?:find|look for)\s+(.+?)\s+(?:on|in)\s+{_URL}\W*$", re.I)
_FOLLOW = re.compile(r"^(?:follow|open|click)\s+(?:link|result)\s*#?\s*(\d+)\W*$", re.I)
_SEARCH = re.compile(r"^(?:please\s+)?(?:search(?: the)? (?:web|internet|online)(?: for)?|google|look up online|web search(?: for)?)\s+(.+?)\W*$", re.I)
_LIVE = re.compile(r"^.*\b(latest news|news (?:about|on|today)|weather (?:in|today|tomorrow|for)|stock price|"
                   r"price of .* (?:today|now)|what'?s happening)\b.*$", re.I)


def _normalize_url(u: str) -> str:
    u = u.rstrip(".,;:!?)\"'")
    return u if re.match(r"https?://", u, re.I) else "https://" + u


class WebSkill(Skill):
    name = "web"
    description = "open web pages, search the web, follow links (optional, needs internet)"

    def __init__(self, fetch: Optional[Callable[[str], Tuple[str, str]]] = None, search_url: str = SEARCH_URL):
        self.fetch = fetch or urllib_fetch
        self.search_url = search_url
        self.last_links: List[Tuple[str, str]] = []

    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip()
        try:
            m = _ASK_PAGE.match(q)
            if m:
                g = [x for x in m.groups() if x]
                url, topic = (g[1], g[0]) if q.lower().startswith(("find", "look for")) else (g[0], g[1])
                return self._ask_page(_normalize_url(url), topic)
            m = _OPEN.match(q)
            if m:
                return self._open(_normalize_url(m.group(1)))
            m = _FOLLOW.match(q)
            if m and self.last_links:
                i = int(m.group(1))
                if not 1 <= i <= len(self.last_links):
                    return Answer(f"There's no link {i}; the last page listed {len(self.last_links)}.", self.name, 0.9, failed=True)
                return self._open(self.last_links[i - 1][1])
            m = _SEARCH.match(q)
            if m:
                return self._search(m.group(1))
            if _LIVE.match(q):
                return self._search(q)
        except Exception as e:  # network errors, timeouts, bad pages: say so, don't guess
            return Answer(f"I couldn't reach the web: {e}", self.name, 0.9, False, [f"{type(e).__name__}: {e}"], failed=True)
        return None

    def _load(self, url: str) -> Page:
        final, body = self.fetch(url)
        page = parse_html(body, final)
        self.last_links = page.links
        return page

    def _open(self, url: str) -> Answer:
        page = self._load(url)
        body = [p for p in page.paragraphs if len(p) >= 40] or page.paragraphs
        summary, size = [], 0
        for p in body:
            if size > 900:
                break
            summary.append(p if len(p) <= 600 else p[:600].rsplit(" ", 1)[0] + " …")
            size += len(summary[-1])
        lines = [f"**{page.title or page.url}**", page.url, ""] + summary
        links = [(t, u) for t, u in page.links if len(t) > 2][:8]
        if links:
            lines += ["", "Links (say \"follow link N\"):"] + [f"  {i}. {t[:70]} — {u}" for i, (t, u) in enumerate(links, 1)]
            self.last_links = links
        return Answer("\n".join(lines), self.name, 0.9, False, [f"fetched {page.url}", f"{len(page.paragraphs)} text blocks"],
                      [page.url])

    def _ask_page(self, url: str, topic: str) -> Answer:
        page = self._load(url)
        index = BM25Index()
        for p in page.paragraphs:
            for s in split_sentences(p):
                if len(s) > 20:
                    index.add(Chunk(s, page.url, ""))
        hits = [c.text for score, c in index.search(topic, k=4) if score > 0]
        want = set(tokenize(topic))
        hits = [h for h in hits if want & set(tokenize(h))]
        if not hits:
            return Answer(f"I read {page.url} but it doesn't seem to say anything about \"{topic}\".", self.name, 0.8, False,
                          [f"fetched {page.url}", "no matching sentence"], [page.url], failed=True)
        text = "\n".join(f"• {h}" for h in hits) + f"\n\n(from {page.title or page.url})"
        return Answer(text, self.name, 0.85, False, [f"fetched {page.url}", f"ranked {len(index)} sentences against \"{topic}\""],
                      [page.url])

    def _search(self, query: str) -> Answer:
        _, body = self.fetch(self.search_url + urllib.parse.quote_plus(query))
        results = parse_search_results(body)[:6]
        if not results:
            return Answer(f"The web search for \"{query}\" came back empty.", self.name, 0.8, failed=True,
                          trace=["no results parsed"])
        self.last_links = [(t, u) for t, u, _ in results]
        lines = [f"Top results for \"{query}\" (say \"follow link N\" to read one):"]
        for i, (t, u, s) in enumerate(results, 1):
            lines.append(f"  {i}. {t} — {u}" + (f"\n     {s[:200]}" if s else ""))
        return Answer("\n".join(lines), self.name, 0.85, False, [f"searched DuckDuckGo for \"{query}\""], [u for _, u, _ in results])
