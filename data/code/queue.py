"""A queue (first in, first out) class with enqueue, dequeue and is_empty.

tags: queue class fifo enqueue dequeue data structure
"""

from collections import deque


class Queue:
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        if not self._items:
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


# tests
q = Queue()
q.enqueue("a")
q.enqueue("b")
assert q.dequeue() == "a" and len(q) == 1
