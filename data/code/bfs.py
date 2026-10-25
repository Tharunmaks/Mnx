"""Breadth-first search over a graph given as an adjacency dict; returns nodes in visit order.

tags: breadth first search bfs graph traversal
"""

from collections import deque


def bfs(graph: dict, start) -> list:
    seen, order, queue = {start}, [], deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order


# tests
assert bfs({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, "a") == ["a", "b", "c", "d"]
