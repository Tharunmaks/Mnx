class Node:
    """A node in a singly linked list."""

    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class LinkedList:
    """A minimal singly linked list."""

    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = node
        self.size += 1

    def to_list(self):
        result = []
        current = self.head
        while current is not None:
            result.append(current.value)
            current = current.next
        return result

    def __len__(self):
        return self.size
