"""Check whether a string looks like a valid email address (simple regex).

tags: validate email address regex
"""

import re


def is_valid_email(s: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s) is not None


# tests
assert is_valid_email("ada@example.com")
assert not is_valid_email("not an email")
