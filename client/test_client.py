import matplotlib.pyplot as plt

from client import Client

from test_utils.utils import print_global_performance, print_token_count
from test_utils.xor import XORData, XORModel

contract_address = input("Enter contract address:\n>")

alice_data = XORData(500).as_dataset()
bob_data, charlie_data = XORData(1000).split()
david_data, eve_data = XORData(1000).split_by_label()

# These clients will evaluate
alice = Client("Alice", alice_data, XORModel, contract_address, 0)

# These clients will train
bob = Client("Bob (good)", bob_data, XORModel, contract_address, 1)
david = Client("David (biased)", david_data, XORModel, contract_address, 3)
eve = Client("Eve (biased)", eve_data, XORModel, contract_address, 4)

TRAINING_ITERATIONS = 16

alice.set_genesis_model()
for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nIteration {i}")
    bob.run_training_round()
    david.run_training_round()
    eve.run_training_round()
    print_global_performance(alice)
    alice.finish_training_round()
    print_token_count(bob)
    print_token_count(david)
    print_token_count(eve)
alice.predict_and_plot()
