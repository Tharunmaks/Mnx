"""Check whether a string is a palindrome, ignoring case and non-alphanumeric characters.

tags: palindrome string reads same forwards backwards
"""

def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]


# tests
assert is_palindrome("A man, a plan, a canal: Panama")
assert not is_palindrome("hello")
assert is_palindrome("")
