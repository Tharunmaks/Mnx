"""A command-line number guessing game.

tags: guess number game random input loop
"""

import random


def guess_number_game(low: int = 1, high: int = 100, read=input) -> int:
    secret = random.randint(low, high)
    tries = 0
    while True:
        tries += 1
        guess = int(read(f"Guess a number between {low} and {high}: "))
        if guess < secret:
            print("Too low")
        elif guess > secret:
            print("Too high")
        else:
            print(f"Correct in {tries} tries!")
            return tries


# tests
import random
random.seed(0)
secret = random.randint(1, 100)
random.seed(0)
guesses = iter([str(secret)])
assert guess_number_game(read=lambda _: next(guesses)) == 1
