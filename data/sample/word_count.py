from collections import Counter


def word_count(text):
    """Return a dict mapping each word to how often it appears."""
    counts = {}
    for word in text.split():
        word = word.lower().strip(".,!?")
        counts[word] = counts.get(word, 0) + 1
    return counts


def most_common_word(text):
    """Return the most frequent word in text."""
    counter = Counter(text.lower().split())
    word, _ = counter.most_common(1)[0]
    return word
