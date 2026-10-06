from dataclasses import dataclass, field
from typing import List


@dataclass
class Task:
    """A todo item."""

    title: str
    done: bool = False
    tags: List[str] = field(default_factory=list)

    def complete(self):
        self.done = True


@dataclass
class TodoList:
    """A collection of tasks."""

    tasks: List[Task] = field(default_factory=list)

    def add(self, title):
        task = Task(title)
        self.tasks.append(task)
        return task

    def pending(self):
        return [t for t in self.tasks if not t.done]
