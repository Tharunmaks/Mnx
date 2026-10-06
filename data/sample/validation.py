import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(address):
    """Return True if address looks like an email."""
    return bool(EMAIL_RE.match(address))


def is_positive_int(value):
    """Return True if value is an int greater than zero."""
    return isinstance(value, int) and value > 0


def require(condition, message):
    """Raise ValueError with message if condition is false."""
    if not condition:
        raise ValueError(message)
