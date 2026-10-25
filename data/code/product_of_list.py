"""Return the product of all numbers in a list; 1 for an empty list.

tags: math, list, product, multiply
"""

def product_of_list(xs):
    total = 1
    for x in xs:
        total *= x
    return total

# tests
assert product_of_list([2, 3, 4]) == 24
assert product_of_list([]) == 1
