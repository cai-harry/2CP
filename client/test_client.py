import pytest

import matplotlib.pyplot as plt

from client import Model, Client, Org

from data import XOR

contract_address = input("Enter contract address:\n>")

alice_data, bob_data = XOR(256).split()
holdout_data = XOR(128).as_dataset()

org = Org(holdout_data, contract_address, 0)
alice = Client(alice_data, 4, contract_address, 1)
bob = Client(bob_data, 4, contract_address, 2)


TRAINING_ITERATIONS = 256
EVALUATE_EVERY = 8
losses = []
losses.append(org.evaluate())
for i in range(TRAINING_ITERATIONS):
    alice.run_train()
    bob.run_train()
    org.run_aggregation()
    if i % EVALUATE_EVERY == EVALUATE_EVERY - 1:
        loss, accuracy = org.evaluate()
        print(f"Iteration {i}\tLoss {loss}\tAccuracy {accuracy}")
        losses.append(loss)
org.predict_and_plot()
