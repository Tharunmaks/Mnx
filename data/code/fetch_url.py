"""Download the text of a web page with the standard library (urllib).

tags: http get request download url web page fetch
"""

from urllib.request import Request, urlopen


def fetch_url(url: str, timeout: float = 10.0) -> str:
    req = Request(url, headers={"User-Agent": "mnx"})
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


# tests
assert callable(fetch_url)
