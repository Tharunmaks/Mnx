"""Read a CSV file into a list of dicts keyed by the header row.

tags: read csv file rows dict parse
"""

import csv


def read_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# tests
import os
import tempfile
p = os.path.join(tempfile.mkdtemp(), "t.csv")
open(p, "w").write("name,age\nAda,36\n")
assert read_csv(p) == [{"name": "Ada", "age": "36"}]
