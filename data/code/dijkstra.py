"""Shortest path distances from a source with Dijkstra's algorithm; graph maps node to [(neighbour, weight)].

tags: dijkstra shortest path graph weighted distances
"""

import heapq


def dijkstra(graph: dict, source) -> dict:
    dist = {source: 0}
    heap = [(0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
            continue
        for nxt, w in graph.get(node, []):
            nd = d + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist


# tests
assert dijkstra({"a": [("b", 1), ("c", 4)], "b": [("c", 2)], "c": []}, "a") == {"a": 0, "b": 1, "c": 3}
