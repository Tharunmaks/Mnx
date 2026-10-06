def caesar_encrypt(text, shift):
    """Shift each letter in text by shift positions."""
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)


def caesar_decrypt(text, shift):
    """Undo a Caesar shift."""
    return caesar_encrypt(text, -shift)
