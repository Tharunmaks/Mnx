"""Return the depth of a binary tree given as nested lists.

tags: tree, binary, depth, recursion, height
"""

def binary_tree_depth(node):
    if not node:
        return 0
    _, left, right = node
    return 1 + max(binary_tree_depth(left), binary_tree_depth(right))

# tests
assert binary_tree_depth(None) == 0
assert binary_tree_depth((1, (2, None, None), None)) == 2
