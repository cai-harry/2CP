import pytest

import matplotlib.pyplot as plt

from client import Client

from data import XOR

contract_address = input("Enter contract address:\n>")

alice_data = XOR(500).as_dataset()
bob_data, charlie_data = XOR(1000).split()
david_data, eve_data = XOR(1000).split_by_label()

# These clients will evaluate
alice = Client("Alice", alice_data, contract_address, 0)

# These clients will train
bob = Client("Bob", bob_data, contract_address, 1)
charlie = Client("Charlie", charlie_data, contract_address, 2)
david = Client("David", david_data, contract_address, 3)
eve = Client("Eve", eve_data, contract_address, 4)

TRAINING_ITERATIONS = 16

def _print_global_performance(client):
    loss, accuracy = client.evaluate_global()
    print(f"Iteration {i}\tLoss {loss}\tAccuracy {accuracy}")

def _print_trainer_performances(client):
    scores = client.evaluate_trainers()
    print(f"Iteration {i}\t\tScores{scores}\t")

def _print_token_count(client):
    tokens, total_tokens = client.get_token_count()
    percent = int(100*tokens/total_tokens)
    print(f"\t{client._name} has {tokens} of {total_tokens} tokens ({percent}%)")

alice.set_genesis_model()
for i in range(1, TRAINING_ITERATIONS+1):
    bob.run_training_round()
    charlie.run_training_round()
    david.run_training_round()
    eve.run_training_round()
    _print_global_performance(alice)
    _print_trainer_performances(alice)
    alice.finish_training_round()
    _print_token_count(alice)
    _print_token_count(bob)
    _print_token_count(charlie)
    _print_token_count(david)
    _print_token_count(eve)
alice.predict_and_plot()
