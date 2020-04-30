import pytest

import matplotlib.pyplot as plt

from client import Client

from data import XOR

contract_address = input("Enter contract address:\n>")

alice_data, bob_data = XOR(1000).split()
charlie_data = XOR(100).as_dataset()

# These clients will train
alice = Client("Alice", alice_data, contract_address, 0)
bob = Client("Bob", bob_data, contract_address, 1)

# These clients will evaluate
charlie = Client("Charlie", charlie_data, contract_address, 2)

TRAINING_ITERATIONS = 64
EVALUATE_EVERY = 4

alice.set_genesis_model()
losses = []
losses.append(charlie.evaluate())
for i in range(TRAINING_ITERATIONS):
    alice.run_training_round()
    bob.run_training_round()
    if i % EVALUATE_EVERY == EVALUATE_EVERY - 1:
        loss, accuracy = charlie.evaluate()
        print(f"Iteration {i}\tLoss {loss}\tAccuracy {accuracy}")
        losses.append(loss)
charlie.predict_and_plot()
