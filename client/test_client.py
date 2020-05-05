import pytest

import matplotlib.pyplot as plt

from client import Client

from data import XOR

contract_address = input("Enter contract address:\n>")

alice_data, bob_data = XOR(1000).split()
charlie_data, david_data = XOR(1000).split_by_label()
google_data = XOR(500).as_dataset()

# These clients will train
alice = Client("Alice", alice_data, contract_address, 0)
bob = Client("Bob", bob_data, contract_address, 1)
charlie = Client("Charlie", charlie_data, contract_address, 2)
david = Client("David", david_data, contract_address, 3)



# These clients will evaluate
google = Client("Google", google_data, contract_address, 4)

TRAINING_ITERATIONS = 16
EVALUATE_EVERY = 1

def print_global_performance(client):
    loss, accuracy = client.evaluate_global()
    print(f"Iteration {i}\tLoss {loss}\tAccuracy {accuracy}")

def print_trainer_performances(client, prev_model):
    scores = client.evaluate_trainers(prev_model)
    print(f"Iteration {i}\t\tScores{scores}\t")

def print_token_count(client):
    tokens, total_tokens = client.get_token_count()
    print(f"\t{client._name} has {tokens} of {total_tokens} tokens")

alice.set_genesis_model()
prev_model = google.get_global_model()
for i in range(1, TRAINING_ITERATIONS+1):
    alice.run_training_round()
    bob.run_training_round()
    # charlie.run_training_round()
    # david.run_training_round()
    if i % EVALUATE_EVERY == 0:
        print_global_performance(google)
        print_trainer_performances(google, prev_model)
        prev_model = google.get_global_model()
google.predict_and_plot()
