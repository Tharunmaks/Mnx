class Rectangle:
    """An axis-aligned rectangle."""

    def __init__(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("dimensions must be non-negative")
        self.width = width
        self.height = height

    def area(self):
        """Return the area of the rectangle."""
        return self.width * self.height

    def perimeter(self):
        """Return the perimeter of the rectangle."""
        return 2 * (self.width + self.height)

    def is_square(self):
        """Return True if width equals height."""
        return self.width == self.height
