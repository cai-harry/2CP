import numpy as np
import matplotlib.pyplot as plt

from client import Client
from utils import print_global_performance, print_token_count

from test_utils.xor import XORData, XORModel, plot_predictions

contract_address = input("Enter contract address:\n>")

alice_data_obj = XORData(500)
alice_data = alice_data_obj.as_dataset()
bob_data, charlie_data = XORData(1000).split()
david_data, eve_data = XORData(1000).split_by_label()

# These clients will evaluate
alice = Client("Alice", alice_data, XORModel, contract_address, 0)

# These clients will train
bob = Client("Bob (unbiased)", bob_data, XORModel, contract_address, 1)
david = Client("David (biased)", david_data, XORModel, contract_address, 3)
eve = Client("Eve (biased)", eve_data, XORModel, contract_address, 4)

TRAINING_ITERATIONS = 3
LEARNING_RATE = 0.3

alice.set_genesis_model()
for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nIteration {i}")
    bob.run_training_round(LEARNING_RATE)
    david.run_training_round(LEARNING_RATE)
    eve.run_training_round(LEARNING_RATE)
    print_global_performance(alice)
    alice.finish_training_round()
    print_token_count(bob)
    print_token_count(david)
    print_token_count(eve)

assert bob.get_token_count() > david.get_token_count()
assert bob.get_token_count() > eve.get_token_count()

pred = alice.predict().squeeze()
alice_pts = np.array(alice_data_obj._data)
plot_predictions(alice_pts, pred)
