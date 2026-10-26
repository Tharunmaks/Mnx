"""Capitalise the first letter of every word in a sentence.

tags: string, title, capitalise, capitalize, words
"""

def title_case(s):
    return " ".join(w.capitalize() for w in s.split())

# tests
assert title_case('hello world') == 'Hello World'
assert title_case('') == ''
