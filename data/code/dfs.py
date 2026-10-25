"""Depth-first search over a graph given as an adjacency dict; returns nodes in visit order.

tags: depth first search dfs graph traversal
"""

def dfs(graph: dict, start) -> list:
    seen, order, stack = set(), [], [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        stack.extend(reversed(graph.get(node, [])))
    return order


# tests
assert dfs({"a": ["b", "c"], "b": ["d"], "c": [], "d": []}, "a") == ["a", "b", "d", "c"]
