"""Format a byte count as a readable string.

tags: bytes, human, format, size, readable
"""

def bytes_to_human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0

# tests
assert bytes_to_human(512) == '512 B'
assert bytes_to_human(2048) == '2.0 KB'
