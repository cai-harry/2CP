import numpy as np
import matplotlib.pyplot as plt
import torch

from context import Client

from test_utils.utils import print_global_performance, print_token_count
from test_utils.xor import XORDataset, XORModel, plot_predictions

def test_integration():

    TRAINING_ITERATIONS = 3
    LEARNING_RATE = 0.3
    TORCH_SEED = 8888

    torch.manual_seed(TORCH_SEED)

    alice_data, alice_targets = XORDataset(500).get()
    bob_data, bob_targets, charlie_data, charlie_targets = XORDataset(1000).split()
    david_data, david_targets, eve_data, eve_targets = XORDataset(1000).split_by_label()

    # These clients will evaluate
    alice = Client("Alice", alice_data, alice_targets, XORModel, 0)

    # These clients will train
    bob = Client("Bob (unbiased)", bob_data, bob_targets, XORModel, 1)
    david = Client("David (biased)", david_data, david_targets, XORModel, 3)
    eve = Client("Eve (biased)", eve_data, eve_targets, XORModel, 4)


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
    alice_pts = np.array(alice_data)
    plot_predictions(alice_pts, pred)
