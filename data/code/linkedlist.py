"""A singly linked list class with append, prepend, to_list and reverse.

tags: linked list class node append reverse data structure
"""

class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def prepend(self, value):
        self.head = Node(value, self.head)

    def reverse(self):
        prev, cur = None, self.head
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        self.head = prev

    def to_list(self) -> list:
        out, cur = [], self.head
        while cur:
            out.append(cur.value)
            cur = cur.next
        return out


# tests
ll = LinkedList()
for v in [1, 2, 3]:
    ll.append(v)
ll.prepend(0)
ll.reverse()
assert ll.to_list() == [3, 2, 1, 0]
