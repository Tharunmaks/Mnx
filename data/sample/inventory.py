class Inventory:
    """Track item quantities in a store."""

    def __init__(self):
        self.items = {}

    def add(self, name, quantity=1):
        self.items[name] = self.items.get(name, 0) + quantity

    def remove(self, name, quantity=1):
        if self.items.get(name, 0) < quantity:
            raise ValueError(f"not enough {name}")
        self.items[name] -= quantity
        if self.items[name] == 0:
            del self.items[name]

    def total(self):
        return sum(self.items.values())
