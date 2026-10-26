"""Turn a number of seconds into hours, minutes and seconds.

tags: time, seconds, hours, minutes, convert, format
"""

def seconds_to_hms(total):
    return total // 3600, (total % 3600) // 60, total % 60

# tests
assert seconds_to_hms(3661) == (1, 1, 1)
assert seconds_to_hms(0) == (0, 0, 0)
