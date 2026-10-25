"""The files agent: search and read your Google Drive (read-only).

Off unless you turn it on (``--tools drive``) and sign in once with
``python -m mnx.agent.tools.google login``. It can't change, share or
delete anything: the only scope it asks for is ``drive.readonly``.

Things it understands::

    recent files in my drive / list my drive
    search my drive for budget / find files about the trip in my drive
    read file 2                        a Google Doc, Sheet (as CSV), Slides or text file
    what does file 2 say about deadlines  the sentences that best answer it
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from mnx.agent.base import Answer, Skill
from mnx.agent.retrieval import BM25Index, Chunk, split_sentences, tokenize
from mnx.agent.tools.google import GoogleClient, NotSignedIn

API = "https://www.googleapis.com/drive/v3/files"
FIELDS = "files(id,name,mimeType,modifiedTime,webViewLink,size)"

#: Google-native files are exported to text; everything else is downloaded if it's text
EXPORT = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
KIND = {"application/vnd.google-apps.document": "Doc", "application/vnd.google-apps.spreadsheet": "Sheet",
        "application/vnd.google-apps.presentation": "Slides", "application/vnd.google-apps.folder": "folder",
        "application/pdf": "PDF"}

_DRIVE = r"(?:google\s+)?drive"
_RECENT = re.compile(rf"^(?:show|list|what'?s in|open)?\s*(?:my\s+)?(?:recent\s+)?(?:files\s+(?:in|on)\s+)?(?:my\s+)?{_DRIVE}(?:\s+files)?\W*$|"
                     rf"^(?:show\s+|list\s+)?(?:my\s+)?recent files(?:\s+(?:in|on)\s+(?:my\s+)?{_DRIVE})?\W*$", re.I)
_SEARCH = re.compile(rf"^(?:search|look through|look in)\s+(?:my\s+)?{_DRIVE}\s+for\s+(.+?)\W*$|"
                     rf"^(?:find|show|list)\s+(?:my\s+)?(?:files|docs|documents|sheets|spreadsheets)\s+(?:about|named|called|with|on)\s+(.+?)"
                     rf"\s+(?:in|on)\s+(?:my\s+)?{_DRIVE}\W*$|"
                     rf"^(?:find|show)\s+(.+?)\s+(?:in|on)\s+(?:my\s+)?{_DRIVE}\W*$", re.I)
_READ = re.compile(r"^(?:read|open|show)\s+(?:drive\s+)?(?:file|doc|document)\s*#?\s*(\d+)\W*$", re.I)
_ASK = re.compile(r"^what does (?:drive\s+)?(?:file|doc|document)\s*#?\s*(\d+)\s+say about\s+(.+?)\W*$", re.I)


def _quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


class DriveSkill(Skill):
    name = "drive"
    description = "search and read your Google Drive, read-only (optional, needs your Google sign-in)"

    def __init__(self, client: Optional[GoogleClient] = None, max_results: int = 10, max_chars: int = 4000):
        self.client = client or GoogleClient()
        self.max_results = max_results
        self.max_chars = max_chars
        self.last: List[Dict] = []

    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip()
        try:
            m = _ASK.match(q)
            if m:
                return self._ask(int(m.group(1)), m.group(2))
            m = _READ.match(q)
            if m and self.last:
                return self._read(int(m.group(1)))
            m = _SEARCH.match(q)
            if m:
                term = next(g for g in m.groups() if g)
                return self._list(f"(name contains '{_quote(term)}' or fullText contains '{_quote(term)}') and trashed = false",
                                  f"files matching \"{term}\"")
            if _RECENT.match(q) and re.search(r"drive|recent files", q, re.I):
                return self._list("trashed = false", "your most recent files")
        except NotSignedIn as e:
            return Answer(str(e), self.name, 0.9, False, ["not signed in to Google"], failed=True)
        except Exception as e:
            return Answer(f"I couldn't reach Google Drive: {e}", self.name, 0.9, False, [f"{type(e).__name__}: {e}"], failed=True)
        return None

    def _list(self, query: str, label: str) -> Answer:
        res = self.client.get(API, q=query, pageSize=self.max_results, orderBy="modifiedTime desc", fields=FIELDS)
        self.last = res.get("files", [])
        if not self.last:
            return Answer(f"No {label} in your Drive.", self.name, 0.9, False, [f"drive query: {query}"])
        lines = [f"{label[0].upper() + label[1:]} (say \"read file N\"):"]
        for n, f in enumerate(self.last, 1):
            kind = KIND.get(f.get("mimeType", ""), f.get("mimeType", "").split("/")[-1])
            lines.append(f"  {n}. {f['name']} ({kind}, {f.get('modifiedTime', '')[:10]})")
        return Answer("\n".join(lines), self.name, 0.9, False, [f"drive query: {query}", f"{len(self.last)} files"],
                      [f.get("webViewLink", "") for f in self.last if f.get("webViewLink")])

    def _text(self, n: int) -> tuple:
        if not 1 <= n <= len(self.last):
            raise LookupError(f"there's no file {n}; the last list had {len(self.last)}")
        f = self.last[n - 1]
        mime = f.get("mimeType", "")
        if mime in EXPORT:
            return f, self.client.request("GET", f"{API}/{f['id']}/export", {"mimeType": EXPORT[mime]}, raw=True)
        if mime.startswith("text/") or mime in ("application/json", "application/xml"):
            return f, self.client.request("GET", f"{API}/{f['id']}", {"alt": "media"}, raw=True)
        return f, None

    def _read(self, n: int) -> Answer:
        try:
            f, text = self._text(n)
        except LookupError as e:
            return Answer(f"I can't open that: {e}.", self.name, 0.9, failed=True)
        link = f.get("webViewLink", "")
        if text is None:
            return Answer(f"\"{f['name']}\" is a {f.get('mimeType')} file, which I can't read as text. Open it here: {link}",
                          self.name, 0.8, False, [f"file {f['id']}"], [link] if link else [], failed=True)
        text = text.strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0] + "\n… (truncated)"
        return Answer(f"**{f['name']}**\n{link}\n\n{text}", self.name, 0.9, False, [f"read file {f['id']}"], [link] if link else [])

    def _ask(self, n: int, topic: str) -> Answer:
        if not self.last:
            return Answer("List some files first, e.g. \"search my drive for budget\".", self.name, 0.9, failed=True)
        try:
            f, text = self._text(n)
        except LookupError as e:
            return Answer(f"I can't open that: {e}.", self.name, 0.9, failed=True)
        if not text:
            return Answer(f"I can't read \"{f['name']}\" as text.", self.name, 0.8, failed=True)
        index = BM25Index()
        for line in text.splitlines():
            for s in split_sentences(line):
                if len(s) > 15:
                    index.add(Chunk(s, f["name"], ""))
        want = set(tokenize(topic))
        hits = [c.text for score, c in index.search(topic, k=4) if score > 0 and want & set(tokenize(c.text))]
        if not hits:
            return Answer(f"\"{f['name']}\" doesn't seem to say anything about \"{topic}\".", self.name, 0.8, failed=True)
        return Answer("\n".join(f"• {h}" for h in hits) + f"\n\n(from {f['name']})", self.name, 0.85, False,
                      [f"ranked {len(index)} sentences of {f['name']}"], [f.get("webViewLink", "")])
