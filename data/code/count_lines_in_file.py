"""Count the lines in a text file.

tags: count lines file wc
"""

def count_lines_in_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


# tests
import os
import tempfile
p = os.path.join(tempfile.mkdtemp(), "t.txt")
open(p, "w").write("1\n2\n3\n")
assert count_lines_in_file(p) == 3
