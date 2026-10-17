"""A Bee's text-mode browser: opens web pages with plain HTTP, reads them, follows links and fills in forms.

Used where a full browser can't run (a phone in Termux has no Chromium for the Bees): it works anywhere, for reading,
searching and simple sites. It doesn't run the page's JavaScript, so app-like sites (shops, logins) need the full
browser; the Bee says so when a page comes back empty. Private and local addresses are never opened.
"""

from __future__ import annotations

import asyncio
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlparse

import httpx

UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36"
MAX_BYTES = 2_500_000
WINDOW = 2500
transport: httpx.AsyncBaseTransport | None = None  # tests use a fake web


class ReaderError(RuntimeError):
    pass


class _Page(HTMLParser):
    """Title, readable text, links and forms of one HTML page."""
    SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "header", "footer", "nav", "ul", "ol", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.text, self.links, self.forms = "", [], [], []
        self._skip = 0
        self._in_title = False
        self._link: dict | None = None
        self._form: dict | None = None
        self._button: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.text.append("\n")
        if tag == "a" and a.get("href") and not a["href"].startswith(("javascript:", "#", "mailto:", "tel:")):
            self._link = {"href": a["href"], "text": a.get("aria-label") or a.get("title") or ""}
        if tag == "form":
            self._form = {"action": a.get("action", ""), "method": (a.get("method") or "get").lower(), "fields": []}
            self.forms.append(self._form)
        if self._form is not None and tag in ("input", "textarea", "select"):
            kind = (a.get("type") or ("textarea" if tag == "textarea" else "text")).lower()
            if a.get("name") and kind not in ("file", "image", "reset"):
                self._form["fields"].append({"name": a["name"], "type": kind, "value": a.get("value", ""),
                                             "label": a.get("placeholder") or a.get("aria-label") or a.get("title") or a["name"]})
        if self._form is not None and tag == "button":
            self._button = {"label": a.get("aria-label") or "", "name": a.get("name", ""), "value": a.get("value", "")}

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._link:
            self._link["text"] = " ".join(self._link["text"].split())[:80]
            if self._link["text"]:
                self.links.append(self._link)
            self._link = None
        if tag == "form":
            self._form = None
        if tag == "button" and self._button is not None and self._form is not None:
            self._form["fields"].append({"name": self._button["name"], "type": "submit", "value": self._button["value"],
                                         "label": " ".join(self._button["label"].split())[:60] or "Submit"})
            self._button = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._skip:
            return
        self.text.append(data)
        if self._link is not None:
            self._link["text"] += data
        if self._button is not None:
            self._button["label"] += data

    def readable(self) -> str:
        t = unescape("".join(self.text))
        lines = [" ".join(ln.split()) for ln in t.splitlines()]
        return "\n".join(ln for ln in lines if ln)


class ReaderSurface:
    """Same moves as the Bee's full browser (hive/apps.py WebSurface), over plain HTTP."""
    kind = "web"
    reader = True

    def __init__(self):
        self.url, self.title, self.text = "", "", ""
        self.links: list[dict] = []
        self.forms: list[dict] = []
        self.els: list[dict] = []
        self.offset = 0
        self.history: list[str] = []
        self.cookies = httpx.Cookies()  # a session's cookies, so a site remembers you between pages

    async def _fetch(self, method: str, url: str, data: dict | None = None) -> None:
        from .browser import _blocked_host
        if urlparse(url).scheme not in ("http", "https"):
            raise ReaderError("Only http and https pages can be opened")
        if await asyncio.to_thread(_blocked_host, urlparse(url).hostname or ""):
            raise ReaderError("That address is on a private network; Bees don't open those")
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True, transport=transport, cookies=self.cookies,
                                         headers={"User-Agent": UA, "Accept-Language": "en-IN,en;q=0.9"}) as c:
                async with c.stream(method, url, params=data if method == "GET" else None, data=data if method == "POST" else None) as r:
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            break
                    self.cookies.update(r.cookies)
                    final, status, ctype = str(r.url), r.status_code, r.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            raise ReaderError(explain_http(exc)) from exc
        if status >= 400:
            raise ReaderError(f"The site answered {status}" + (" (it doesn't allow automated visits)" if status in (401, 403, 429) else ""))
        if self.url:
            self.history.append(self.url)
        self.url, self.offset = final, 0
        if "html" not in ctype and "text" not in ctype:
            self.title, self.text, self.links, self.forms = final, f"(a {ctype or 'binary'} file, {len(body):,} bytes)", [], []
            return
        p = _Page()
        p.feed(body.decode(errors="replace"))
        self.title = " ".join(p.title.split()) or final
        self.text = p.readable()
        self.links = [{"href": urljoin(final, a["href"]), "text": a["text"]} for a in p.links][:200]
        self.forms = [{**f, "action": urljoin(final, f["action"] or final)} for f in p.forms][:10]

    async def open(self, url: str) -> None:
        await self._fetch("GET", url if "://" in url else "https://" + url)

    async def goto(self, url: str) -> None:
        await self.open(url)

    async def observe(self) -> dict:
        self.els = []
        for fi, f in enumerate(self.forms):
            for field in f["fields"]:
                if field["type"] in ("hidden",):
                    continue
                kind = "button" if field["type"] == "submit" else f"input:{field['type']}"
                self.els.append({"kind": kind, "label": field["label"], "form": fi, "field": field})
        for a in self.links[:120]:
            self.els.append({"kind": "a", "label": a["text"], "href": a["href"]})
        window = self.text[self.offset:self.offset + WINDOW]
        more = len(self.text) - self.offset - WINDOW
        if more > 0:
            window += f"\n… {more:,} more characters below (scroll)"
        if len(self.text.strip()) < 80 and not self.forms:
            window += "\n(This page shows almost nothing without JavaScript: it needs the full browser.)"
        return {"place": self.url, "title": self.title, "text": window, "elements": [{"label": e["label"], "kind": e["kind"]} for e in self.els]}

    def _el(self, i) -> dict:
        try:
            return self.els[int(i)]
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"There's no element {i}") from exc

    async def click(self, i) -> None:
        e = self._el(i)
        if e["kind"] == "a":
            await self.open(e["href"])
        elif e["kind"] == "button":
            await self._submit(e["form"], e["field"])
        else:
            self._focus = e

    async def type(self, i, text: str, enter: bool = False) -> None:
        e = self._el(i) if i is not None else getattr(self, "_focus", None)
        if not e or "field" not in e:
            raise ValueError("That isn't a box to type into")
        e["field"]["value"] = text
        self._focus = e
        if enter:
            await self._submit(e["form"])

    async def press(self, key: str) -> None:
        if key.lower() == "enter" and getattr(self, "_focus", None):
            await self._submit(self._focus["form"])

    async def _submit(self, form_index: int, button: dict | None = None) -> None:
        f = self.forms[form_index]
        data = {x["name"]: x["value"] for x in f["fields"] if x["name"] and x["type"] not in ("submit",)}
        if button and button.get("name"):
            data[button["name"]] = button.get("value", "")
        await self._fetch("POST" if f["method"] == "post" else "GET", f["action"], data)

    async def scroll(self, down: bool = True) -> None:
        self.offset = max(0, min(self.offset + (WINDOW if down else -WINDOW), max(0, len(self.text) - 200)))

    async def back(self) -> None:
        if self.history:
            url = self.history.pop()
            await self._fetch("GET", url)
            self.history.pop()  # _fetch pushed the page we left

    async def shot(self) -> tuple[bytes, str]:
        return b"", ""  # text mode: no picture

    def summary(self, chars: int = 600) -> str:
        return self.text[:chars]


def explain_http(exc: Exception) -> str:
    """What went wrong opening a page, in words."""
    msg = str(exc)
    if isinstance(exc, httpx.TimeoutException) or "timeout" in msg.lower():
        return "The site took too long to answer"
    if isinstance(exc, httpx.ConnectError) or "Name or service not known" in msg or "nodename" in msg:
        return "Couldn't reach the site (check the address, or this machine has no internet)"
    if "TooManyRedirects" in type(exc).__name__:
        return "The site kept redirecting"
    return f"The page didn't load: {msg[:150]}"


def explain_browser(msg: str) -> str:
    """Chromium's error codes, in words."""
    for code, words in (("ERR_NAME_NOT_RESOLVED", "that address doesn't exist (no such website)"),
                        ("ERR_INTERNET_DISCONNECTED", "this machine has no internet right now"),
                        ("ERR_TUNNEL_CONNECTION_FAILED", "this machine isn't allowed to reach that site (a proxy or firewall blocks it)"),
                        ("ERR_PROXY_CONNECTION_FAILED", "this machine's internet proxy isn't answering"),
                        ("ERR_CONNECTION_REFUSED", "the site refused the connection"),
                        ("ERR_CONNECTION_TIMED_OUT", "the site took too long to answer"),
                        ("ERR_CERT", "the site's security certificate isn't valid"),
                        ("Timeout", "the page took too long to load"),
                        ("blockedbyclient", "that address is on a private network; Bees don't open those")):
        if code.lower() in msg.lower():
            return words
    return msg[:200]
