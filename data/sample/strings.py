def reverse_string(s):
    """Return s reversed."""
    return s[::-1]


def is_palindrome(s):
    """Return True if s reads the same forwards and backwards."""
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def count_vowels(s):
    """Return the number of vowels in s."""
    return sum(1 for c in s.lower() if c in "aeiou")


def capitalize_words(s):
    """Capitalize the first letter of every word."""
    return " ".join(word.capitalize() for word in s.split())
