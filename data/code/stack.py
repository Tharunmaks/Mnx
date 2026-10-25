"""A stack (last in, first out) class with push, pop, peek and is_empty.

tags: stack class lifo push pop data structure
"""

class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        return self._items[-1] if self._items else None

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


# tests
s = Stack()
s.push(1)
s.push(2)
assert s.pop() == 2 and s.peek() == 1 and len(s) == 1
