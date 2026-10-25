"""Check that a password is long enough and mixes letters and digits.

tags: string, password, validate, check, strength
"""

def validate_password(p):
    return len(p) >= 8 and any(c.isdigit() for c in p) and any(c.isalpha() for c in p)

# tests
assert validate_password('abcd1234')
assert not validate_password('abcd')
