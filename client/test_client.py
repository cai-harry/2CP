import pytest

import matplotlib.pyplot as plt

from client import Model, Client, Org

from mock_contract import MockContract
from data import XOR

mock_contract = MockContract()


alice_data, bob_data = XOR(256).split()
holdout_data = XOR(128).as_dataset()

org = Org(holdout_data, mock_contract)
alice = Client(alice_data, 4, mock_contract)
bob = Client(bob_data, 4, mock_contract)


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
