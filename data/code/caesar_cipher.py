"""Encrypt text with a Caesar cipher shifting letters by k (use -k to decrypt).

tags: caesar cipher encrypt shift letters
"""

def caesar_cipher(text: str, k: int) -> str:
    out = []
    for c in text:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base + k) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


# tests
assert caesar_cipher("Hello, World", 3) == "Khoor, Zruog"
assert caesar_cipher(caesar_cipher("abc", 5), -5) == "abc"
