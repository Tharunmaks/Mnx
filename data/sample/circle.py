import math


class Circle:
    """A circle defined by its radius."""

    def __init__(self, radius):
        self.radius = radius

    def area(self):
        """Return the area of the circle."""
        return math.pi * self.radius ** 2

    def circumference(self):
        """Return the circumference of the circle."""
        return 2 * math.pi * self.radius
