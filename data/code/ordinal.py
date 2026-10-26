"""Turn a number into its ordinal form, such as 1st or 22nd.

tags: string, ordinal, number, suffix, format
"""

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# tests
assert ordinal(1) == '1st'
assert ordinal(12) == '12th'
assert ordinal(22) == '22nd'
