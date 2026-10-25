"""Convert a temperature from Celsius to Fahrenheit.

tags: celsius fahrenheit temperature convert
"""

def celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


# tests
assert celsius_to_fahrenheit(100) == 212
assert celsius_to_fahrenheit(-40) == -40
