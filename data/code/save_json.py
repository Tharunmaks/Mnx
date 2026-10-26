"""Save a Python object to a JSON file and load it back.

tags: json save load file write read
"""

import json


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# tests
import os
import tempfile
p = os.path.join(tempfile.mkdtemp(), "t.json")
save_json({"a": [1, 2]}, p)
assert load_json(p) == {"a": [1, 2]}
