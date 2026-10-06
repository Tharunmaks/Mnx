class Counter:
    """A counter that can be incremented and reset."""

    def __init__(self, start=0):
        self.value = start

    def increment(self, step=1):
        self.value += step
        return self.value

    def decrement(self, step=1):
        self.value -= step
        return self.value

    def reset(self):
        self.value = 0
