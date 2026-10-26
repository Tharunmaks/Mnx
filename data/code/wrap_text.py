"""Break a sentence into lines of at most n characters, on word boundaries.

tags: string, wrap, text, lines, width, words
"""

def wrap_text(s, width):
    words = s.split()
    lines, line = [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines

# tests
assert wrap_text('a bb ccc', 4) == ['a bb', 'ccc']
assert wrap_text('', 5) == []
