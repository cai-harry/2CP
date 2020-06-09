import filecmp

import numpy as np
import matplotlib.pyplot as plt
import torch

from client import Client
from utils import print_global_performance, print_token_count

from test_utils.xor import XORDataset, XORModel, plot_predictions


def test_integration():

    TRAINING_ITERATIONS = 2
    TRAINING_HYPERPARAMS = {
        'epochs': 2,
        'learning_rate': 0.3
    }
    TORCH_SEED = 8888

    torch.manual_seed(TORCH_SEED)

    alice_data, alice_targets = XORDataset(256).get()
    bob_data, bob_targets, charlie_data, charlie_targets = XORDataset(
        128).split()
    david_data, david_targets, eve_data, eve_targets = XORDataset(4).split()

    # These clients will evaluate
    alice = Client("Alice", alice_data, alice_targets, XORModel, 0)

    # These clients will train
    bob = Client("Bob", bob_data, bob_targets, XORModel, 1)
    charlie = Client("Charlie", charlie_data, charlie_targets, XORModel, 2)
    david = Client("David", david_data, david_targets, XORModel, 3)
    eve = Client("Eve", eve_data, eve_targets, XORModel, 4)

    print("Alice setting genesis...")
    genesis_tx = alice.set_genesis_model()
    pending_txs = [genesis_tx]
    alice.wait_for(pending_txs)
    print("Alice done and transaction confirmed")

    # Training
    for i in range(1, TRAINING_ITERATIONS+1):
        print(f"\nIteration {i}")
        print("\tBob training...")
        tx_b = bob.run_training_round(**TRAINING_HYPERPARAMS)
        print("\tCharlie training...")
        tx_c = charlie.run_training_round(**TRAINING_HYPERPARAMS)
        print("\tDavid training...")
        tx_d = david.run_training_round(**TRAINING_HYPERPARAMS)
        print("\tEve training...")
        tx_e = eve.run_training_round(**TRAINING_HYPERPARAMS)
        pending_txs.extend([tx_b, tx_c, tx_d, tx_e])
        print("\tAlice waiting for others' txs...")
        alice.wait_for(pending_txs)
        print("\tAlice evaluating global...")
        print_global_performance(alice)
    
    # Retrospective evaluation
    for i in range(1, TRAINING_ITERATIONS+1):
        print(f"\nEvaluating iteration {i}")
        print("\tAlice calculating SVs...")
        scores = alice.evaluate_updates(i)
        print("\tAlice setting SVs...")
        txs = alice.set_tokens(scores)
        pending_txs.extend(txs)

    print("\tAlice waiting for her txs to finish...")
    alice.wait_for(pending_txs)
    print_token_count(bob)
    print_token_count(charlie)
    print_token_count(david)
    print_token_count(eve)

    assert bob.get_token_count() > david.get_token_count(
    ), "Bob ended up with fewer tokens than David"
    assert bob.get_token_count() > eve.get_token_count(
    ), "Bob ended up with fewer tokens than Eve"
    assert charlie.get_token_count() > david.get_token_count(
    ), "Charlie ended up with fewer tokens than David"
    assert charlie.get_token_count() > eve.get_token_count(
    ), "Charlie ended up with fewer tokens than Eve"
    
    alice_global_model, _ = alice.get_current_global_model()
    bob_global_model, _ = bob.get_current_global_model()

    assert _same_weights(
        alice_global_model,
        bob_global_model
    ), "Alice and Bob ran the same aggregation but got different model weights"

    assert str(alice_global_model.state_dict()) == \
        str(bob_global_model.state_dict()), \
            "Alice and Bob ran the same aggregation but got different model dicts"

def _same_weights(model_a, model_b):
    """
    Checks if two pytorch models have the same weights.
    https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    """
    for params_a, params_b in zip(model_a.parameters(), model_b.parameters()):
        if (params_a.data!=params_b.data).sum() > 0:
            return False
    return True
