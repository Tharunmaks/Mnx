"""Read a text file and return its lines without trailing newlines.

tags: read file lines text open
"""

def read_lines(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# tests
import os
import tempfile
p = os.path.join(tempfile.mkdtemp(), "t.txt")
open(p, "w").write("a\nb\n")
assert read_lines(p) == ["a", "b"]
