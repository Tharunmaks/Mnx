"""Check whether a year is a leap year in the Gregorian calendar.

tags: leap year calendar
"""

def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# tests
assert is_leap_year(2000) and not is_leap_year(1900) and is_leap_year(2024)
