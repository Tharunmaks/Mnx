"""Count how often each word appears in a text, case-insensitively.

tags: word count frequency dictionary text words
"""

import re
from collections import Counter


def word_count(text: str) -> dict:
    return dict(Counter(re.findall(r"[a-z0-9']+", text.lower())))


# tests
assert word_count("the cat and the hat") == {"the": 2, "cat": 1, "and": 1, "hat": 1}
