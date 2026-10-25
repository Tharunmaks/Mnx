"""A simple calculator function for +, -, *, / on two numbers.

tags: calculator add subtract multiply divide operator
"""

def simple_calculator(a: float, op: str, b: float) -> float:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b
    raise ValueError(f"unknown operator {op!r}")


# tests
assert simple_calculator(6, "/", 3) == 2 and simple_calculator(2, "*", 4) == 8
