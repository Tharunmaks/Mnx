"""Browser Bee engine: a real Chromium the Hive (and you) can drive and watch.

Every action returns a fresh screenshot so the web app's Watch view stays live.
Private and local addresses are blocked, so a web page can't steer the browser
into your server or home network.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

VIEWPORT = {"width": 1280, "height": 800}
MAX_TEXT = 20000


def _blocked_host(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if not host or host == "localhost" or host.endswith(".localhost") or host.endswith(".internal"):
        return True
    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            ips = [ipaddress.ip_address(i[4][0]) for i in socket.getaddrinfo(host, None)]
        except OSError:
            return False  # let the browser report the DNS failure
    return any(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast for ip in ips)


class BrowserError(RuntimeError):
    pass


class BrowserSession:
    def __init__(self, profile_dir: str | os.PathLike | None = None) -> None:
        # With a profile dir the browser keeps its cookies, logins and site storage on disk,
        # so a Bee stays signed in across restarts.
        self.profile_dir = profile_dir
        self._context = None
        self.page = None
        self._lock = asyncio.Lock()
        self._host_cache: dict[str, bool] = {}
        self.last_used = time.monotonic()

    @property
    def running(self) -> bool:
        return self.page is not None and not self.page.is_closed()

    async def _ensure(self) -> None:
        self.last_used = time.monotonic()
        if self.running:
            return
        if self._context:
            await self._close()
        pw = await _playwright()
        kw = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"],
              "viewport": VIEWPORT, "locale": "en-IN"}
        if os.getenv("MNX_CHROMIUM_PATH"):
            kw["executable_path"] = os.environ["MNX_CHROMIUM_PATH"]
        if os.getenv("MNX_BROWSER_PROXY"):
            kw["proxy"] = {"server": os.environ["MNX_BROWSER_PROXY"]}
        try:
            if self.profile_dir:
                os.makedirs(self.profile_dir, exist_ok=True)
                self._context = await pw.chromium.launch_persistent_context(str(self.profile_dir), **kw)
            else:
                viewport, locale = kw.pop("viewport"), kw.pop("locale")
                browser = await pw.chromium.launch(**kw)
                self._context = await browser.new_context(viewport=viewport, locale=locale)
        except Exception as exc:
            raise BrowserError(f"Couldn't start Chromium ({str(exc).splitlines()[0]}). Run 'playwright install chromium' or set MNX_CHROMIUM_PATH.") from exc
        await self._context.route("**/*", self._guard)
        pages = self._context.pages
        self.page = pages[0] if pages else await self._context.new_page()
        self._context.on("page", self._adopt_popup)

    async def _adopt_popup(self, popup) -> None:
        # Keep one visible tab: follow pages that open in a new window.
        self.page = popup

    async def _guard(self, route) -> None:
        url = urlparse(route.request.url)
        if url.scheme in ("data", "blob", "about"):
            return await route.continue_()
        if url.scheme not in ("http", "https"):
            return await route.abort("blockedbyclient")
        host = url.hostname or ""
        if host not in self._host_cache:
            self._host_cache[host] = await asyncio.to_thread(_blocked_host, host)
        if self._host_cache[host]:
            return await route.abort("blockedbyclient")
        await route.continue_()

    # ----- actions (each returns the current state + screenshot) -----
    async def do(self, action: str, **a) -> dict:
        self.last_used = time.monotonic()
        async with self._lock:
            if action == "close":
                await self._close()
                return {"running": False}
            await self._ensure()
            p = self.page
            try:
                if action == "goto":
                    url = str(a.get("url", "")).strip()
                    if not url:
                        raise BrowserError("No URL")
                    if "://" not in url:
                        url = "https://" + url
                    if urlparse(url).scheme not in ("http", "https"):
                        raise BrowserError("Only http and https pages can be opened")
                    await p.goto(url, wait_until="domcontentloaded", timeout=30000)
                elif action == "click":
                    if a.get("text"):
                        await p.get_by_text(str(a["text"]), exact=False).first.click(timeout=8000)
                    elif a.get("selector"):
                        await p.click(str(a["selector"]), timeout=8000)
                    else:
                        await p.mouse.click(float(a["x"]), float(a["y"]))
                elif action == "type":
                    if a.get("selector"):
                        await p.fill(str(a["selector"]), str(a.get("text", "")), timeout=8000)
                    else:
                        await p.keyboard.type(str(a.get("text", "")), delay=15)
                elif action == "press":
                    await p.keyboard.press(str(a.get("key", "Enter")))
                elif action == "scroll":
                    await p.mouse.wheel(0, float(a.get("dy", 600)))
                elif action == "back":
                    await p.go_back(timeout=15000)
                elif action == "forward":
                    await p.go_forward(timeout=15000)
                elif action == "reload":
                    await p.reload(timeout=30000)
                elif action == "read":
                    text = await p.inner_text("body", timeout=8000)
                    return {**await self.state(shot=False), "text": text[:MAX_TEXT]}
                elif action == "elements":
                    return {**await self.state(shot=False), "elements": await self._elements()}
                elif action != "screenshot":
                    raise BrowserError(f"Unknown action: {action}")
                await p.wait_for_timeout(350)
            except BrowserError:
                raise
            except Exception as exc:
                raise BrowserError(str(exc).split("\n")[0][:300]) from exc
            return await self.state()

    async def state(self, shot: bool = True) -> dict:
        if not self.running:
            return {"running": False}
        p = self.page
        out = {"running": True, "url": p.url, "title": await p.title(), **VIEWPORT}
        if shot:
            img = await p.screenshot(type="jpeg", quality=60)
            out["image"] = "data:image/jpeg;base64," + base64.b64encode(img).decode()
        return out

    async def _elements(self) -> list[dict]:
        """Visible clickable / typeable elements with their centre point — what a Bee 'sees'."""
        return await self.page.evaluate("""() => {
          const sel = 'a,button,input,textarea,select,[role=button],[role=link],[onclick],[contenteditable=true]';
          const out = [];
          for (const el of document.querySelectorAll(sel)) {
            const r = el.getBoundingClientRect();
            if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.top > innerHeight) continue;
            const label = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || el.title || '').trim().slice(0, 80);
            out.push({tag: el.tagName.toLowerCase(), type: el.type || null, label,
                      x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2)});
            if (out.length >= 150) break;
          }
          return out;
        }""")

    async def _close(self) -> None:
        ctx, self._context, self.page = self._context, None, None
        if ctx:
            browser = ctx.browser
            try:
                await ctx.close()
                if browser and not self.profile_dir:
                    await browser.close()
            except Exception:
                pass

    async def shutdown(self) -> None:
        async with self._lock:
            await self._close()


_pw = None
_pw_lock = asyncio.Lock()


async def _playwright():
    global _pw
    async with _pw_lock:
        if _pw is None:
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise BrowserError("Playwright isn't installed: pip install playwright && playwright install chromium") from exc
            _pw = await async_playwright().start()
        return _pw


async def stop_playwright() -> None:
    global _pw
    if _pw:
        await _pw.stop()
        _pw = None
