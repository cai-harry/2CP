import pytest

import matplotlib.pyplot as plt

from client import Client

from data import XOR

contract_address = input("Enter contract address:\n>")

alice_data, bob_data = XOR(256).split()
charlie_data = XOR(128).as_dataset()

# These clients will train
alice = Client(alice_data, 4, contract_address, 0)
bob = Client(bob_data, 4, contract_address, 1)

# These clients will evaluate
charlie = Client(charlie_data, 4, contract_address, 2)

TRAINING_ITERATIONS = 256
EVALUATE_EVERY = 8

# alice.set_genesis_model()
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
