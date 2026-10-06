from collections import deque


class Queue:
    """A simple first-in first-out queue."""

    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.items.popleft()

    def is_empty(self):
        return len(self.items) == 0

    def __len__(self):
        return len(self.items)
