class InsufficientFunds(Exception):
    """Raised when a withdrawal exceeds the balance."""


class BankAccount:
    """A bank account with deposits and withdrawals."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFunds("not enough money")
        self.balance -= amount
        return self.balance
