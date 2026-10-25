"""A tiny to-do list manager class with add, complete and pending.

tags: todo list task manager class
"""

class TodoList:
    def __init__(self):
        self.tasks = []

    def add(self, title: str) -> int:
        self.tasks.append({"title": title, "done": False})
        return len(self.tasks) - 1

    def complete(self, index: int) -> None:
        self.tasks[index]["done"] = True

    def pending(self) -> list:
        return [t["title"] for t in self.tasks if not t["done"]]


# tests
t = TodoList()
i = t.add("write code")
t.add("test code")
t.complete(i)
assert t.pending() == ["test code"]
