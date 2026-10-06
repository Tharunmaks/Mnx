class TreeNode:
    """A node of a binary search tree."""

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def insert(root, value):
    """Insert value into the tree rooted at root and return the root."""
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)
    return root


def inorder(root):
    """Return the values of the tree in sorted order."""
    if root is None:
        return []
    return inorder(root.left) + [root.value] + inorder(root.right)


def contains(root, value):
    """Return True if value is in the tree."""
    while root is not None:
        if value == root.value:
            return True
        root = root.left if value < root.value else root.right
    return False
