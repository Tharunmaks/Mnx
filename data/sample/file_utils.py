import os


def read_lines(path):
    """Return the lines of a text file without trailing newlines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    """Write lines to a text file, one per line."""
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def file_size(path):
    """Return the size of a file in bytes."""
    return os.path.getsize(path)
