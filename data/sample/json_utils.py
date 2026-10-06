import json


def load_json(path):
    """Load a JSON document from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data, indent=2):
    """Write data to a file as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def pretty(data):
    """Return data as a formatted JSON string."""
    return json.dumps(data, indent=2, sort_keys=True)
