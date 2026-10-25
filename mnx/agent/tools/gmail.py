"""The mail agent: read and search your Gmail, and save drafts.

Off unless you turn it on (``--tools gmail``) and sign in once with
``python -m mnx.agent.tools.google login``. It never sends mail: "send"
and "draft" both save a draft you can review and send from Gmail. Email
text is shown to you, never followed as instructions.

Things it understands::

    check my email / show my inbox / any unread emails?
    emails from alice / search my email for invoice / find emails about the trip
    read email 2                      the full text of a message from the last list
    draft an email to bob@x.com subject Lunch saying see you at 1
    draft a reply to email 2 saying thanks, sounds good
"""

from __future__ import annotations

import base64
import html
import re
from email.message import EmailMessage
from typing import Dict, List, Optional

from mnx.agent.base import Answer, Skill
from mnx.agent.tools.google import GoogleClient, NotSignedIn
from mnx.agent.tools.web import parse_html

API = "https://gmail.googleapis.com/gmail/v1/users/me"

_MAIL = r"(?:e-?mails?|mail|inbox|gmail|messages?)"
_LIST = re.compile(rf"^(?:please\s+)?(?:check|show|list|open|read|get|any|do i have(?: any)?)\s+(?:me\s+)?(?:my\s+)?(new\s+|unread\s+|latest\s+|recent\s+)?{_MAIL}\W*$|"
                   rf"^(?:what'?s|whats) in my {_MAIL}\W*$|^(?:any|do i have(?: any)?) (new|unread) {_MAIL}\W*$", re.I)
_FROM = re.compile(rf"^(?:show\s+|find\s+|list\s+)?(?:my\s+)?(?:unread\s+)?{_MAIL}\s+from\s+(.+?)\W*$", re.I)
_SEARCH = re.compile(rf"^(?:search|look through|look in)\s+(?:my\s+)?{_MAIL}\s+for\s+(.+?)\W*$|"
                     rf"^(?:find|show)\s+(?:my\s+)?{_MAIL}\s+(?:about|with|mentioning|containing)\s+(.+?)\W*$", re.I)
_READ = re.compile(rf"^(?:read|open|show)\s+(?:e-?mail|message|mail)\s*#?\s*(\d+)\W*$", re.I)
_DRAFT = re.compile(r"^(?:draft|write|compose|send)\s+(?:an?\s+)?(?:e-?mail|mail|message)\s+to\s+(\S+@\S+?)"
                    r"(?:\s+(?:with\s+)?subject\s+(.+?))?\s+(?:saying|that says|with body|body)\s*:?\s*(.+)$", re.I | re.S)
_REPLY = re.compile(r"^(?:draft|write|send)\s+(?:an?\s+)?reply\s+to\s+(?:e-?mail|message)\s*#?\s*(\d+)\s+(?:saying|that says)\s*:?\s*(.+)$",
                    re.I | re.S)


def _header(msg: Dict, name: str) -> str:
    for h in msg.get("payload", {}).get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _b64(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def message_text(payload: Dict) -> str:
    """Plain text of a Gmail payload: text/plain if present, else stripped HTML."""
    plain: List[str] = []
    htmls: List[str] = []

    def walk(part):
        mime, data = part.get("mimeType", ""), part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(_b64(data))
        elif data and mime == "text/html":
            htmls.append(_b64(data))
        for p in part.get("parts", []) or []:
            walk(p)

    walk(payload)
    if plain:
        return "\n".join(plain).strip()
    if htmls:
        return "\n\n".join(parse_html(htmls[0], "").paragraphs)
    return ""


class GmailSkill(Skill):
    name = "gmail"
    description = "read and search your Gmail, save drafts (optional, needs your Google sign-in)"

    def __init__(self, client: Optional[GoogleClient] = None, max_results: int = 8):
        self.client = client or GoogleClient()
        self.max_results = max_results
        self.last: List[Dict] = []  # messages from the last list, for "read email N"

    def answer(self, question: str) -> Optional[Answer]:
        q = question.strip()
        try:
            m = _REPLY.match(q)
            if m:
                return self._reply(int(m.group(1)), m.group(2).strip())
            m = _DRAFT.match(q)
            if m:
                return self._draft(m.group(1).rstrip(".,;"), (m.group(2) or "").strip(), m.group(3).strip(),
                                   asked_to_send=q.lower().startswith("send"))
            m = _READ.match(q)
            if m:
                return self._read(int(m.group(1)))
            m = _FROM.match(q)
            if m:
                who = m.group(1)
                return self._list(f"from:{who}" + (" is:unread" if "unread" in q.lower() else ""), f"emails from {who}")
            m = _SEARCH.match(q)
            if m:
                term = m.group(1) or m.group(2)
                return self._list(term, f"emails matching \"{term}\"")
            m = _LIST.match(q)
            if m:
                unread = bool(re.search(r"\b(new|unread)\b", q, re.I))
                return self._list("in:inbox" + (" is:unread" if unread else ""), "unread emails" if unread else "your inbox")
        except NotSignedIn as e:
            return Answer(str(e), self.name, 0.9, False, ["not signed in to Google"], failed=True)
        except Exception as e:
            return Answer(f"I couldn't reach Gmail: {e}", self.name, 0.9, False, [f"{type(e).__name__}: {e}"], failed=True)
        return None

    def _list(self, query: str, label: str) -> Answer:
        res = self.client.get(f"{API}/messages", q=query, maxResults=self.max_results)
        ids = [m["id"] for m in res.get("messages", [])]
        if not ids:
            self.last = []
            return Answer(f"No {label} found.", self.name, 0.9, False, [f"gmail query: {query}"])
        msgs = [self.client.get(f"{API}/messages/{i}", format="metadata", metadataHeaders=["From", "Subject", "Date"]) for i in ids]
        self.last = msgs
        lines = [f"{label[0].upper() + label[1:]} ({len(msgs)} shown, say \"read email N\"):"]
        for n, msg in enumerate(msgs, 1):
            unread = "UNREAD" in msg.get("labelIds", [])
            sender = re.sub(r"\s*<[^>]+>", "", _header(msg, "From")) or _header(msg, "From")
            lines.append(f"  {n}. {'● ' if unread else ''}{sender} — {_header(msg, 'Subject') or '(no subject)'}")
            if msg.get("snippet"):
                lines.append(f"     {html.unescape(msg['snippet'])[:140]}")
        return Answer("\n".join(lines), self.name, 0.9, False, [f"gmail query: {query}", f"{len(msgs)} messages"])

    def _pick(self, n: int) -> Dict:
        if not self.last:
            raise LookupError("list some emails first, e.g. \"check my email\"")
        if not 1 <= n <= len(self.last):
            raise LookupError(f"there's no email {n}; the last list had {len(self.last)}")
        return self.last[n - 1]

    def _read(self, n: int) -> Answer:
        try:
            meta = self._pick(n)
        except LookupError as e:
            return Answer(f"I can't open that: {e}.", self.name, 0.9, failed=True)
        msg = self.client.get(f"{API}/messages/{meta['id']}", format="full")
        body = message_text(msg.get("payload", {})) or msg.get("snippet", "")
        if len(body) > 4000:
            body = body[:4000].rsplit(" ", 1)[0] + "\n… (truncated; open it in Gmail for the rest)"
        head = [f"From: {_header(msg, 'From')}", f"Date: {_header(msg, 'Date')}", f"Subject: {_header(msg, 'Subject')}", ""]
        return Answer("\n".join(head) + body, self.name, 0.9, False, [f"read message {meta['id']}"])

    def _save_draft(self, msg: EmailMessage, thread_id: Optional[str] = None) -> Dict:
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        body = {"message": {"raw": raw, **({"threadId": thread_id} if thread_id else {})}}
        return self.client.request("POST", f"{API}/drafts", body=body)

    def _draft(self, to: str, subject: str, text: str, asked_to_send: bool = False) -> Answer:
        msg = EmailMessage()
        msg["To"], msg["Subject"] = to, subject or "(no subject)"
        msg.set_content(text)
        d = self._save_draft(msg)
        note = "I don't send mail myself, so I saved it as a draft. " if asked_to_send else ""
        return Answer(f"{note}Draft saved to {to} with subject \"{msg['Subject']}\". Review and send it from Gmail's Drafts folder.",
                      self.name, 0.9, False, [f"created draft {d.get('id', '?')}"])

    def _reply(self, n: int, text: str) -> Answer:
        try:
            meta = self._pick(n)
        except LookupError as e:
            return Answer(f"I can't reply to that: {e}.", self.name, 0.9, failed=True)
        full = self.client.get(f"{API}/messages/{meta['id']}", format="metadata", metadataHeaders=["From", "Subject", "Message-ID", "Reply-To"])
        msg = EmailMessage()
        msg["To"] = _header(full, "Reply-To") or _header(full, "From")
        subj = _header(full, "Subject")
        msg["Subject"] = subj if subj.lower().startswith("re:") else f"Re: {subj}"
        if _header(full, "Message-ID"):
            msg["In-Reply-To"] = msg["References"] = _header(full, "Message-ID")
        msg.set_content(text)
        d = self._save_draft(msg, full.get("threadId"))
        return Answer(f"Reply draft saved to {msg['To']} (\"{msg['Subject']}\"). Review and send it from Gmail.", self.name, 0.9, False,
                      [f"created draft {d.get('id', '?')} in thread {full.get('threadId', '?')}"])

