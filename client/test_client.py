import pytest

from client import Model, Client, Org

from mock_contract import MockContract
from data import XOR

mock_contract = MockContract()

org = Org(mock_contract)

alice_data, bob_data = XOR(100).split_by_label()

alice = Client(alice_data, mock_contract)
bob = Client(bob_data, mock_contract)

TRAINING_ITERATIONS = 8
for i in range(TRAINING_ITERATIONS):
    alice.run_train()
    bob.run_train()
    org.run_aggregation()

