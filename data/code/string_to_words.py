"""Split a sentence into words on spaces and commas.

tags: string, split, words, sentence, commas
"""

def string_to_words(s):
    return [w for w in s.replace(",", " ").split() if w]

# tests
assert string_to_words('a, b c') == ['a', 'b', 'c']
assert string_to_words('') == []
