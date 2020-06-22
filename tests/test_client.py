import filecmp

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from clients import CrowdsourceClient, ConsortiumSetupClient, ConsortiumClient
from utils import print_global_performance, print_token_count

from test_utils.xor import XORDataset, XORModel, plot_predictions

TRAINING_ITERATIONS = 2
TRAINING_HYPERPARAMS = {
    'epochs': 2,
    'learning_rate': 0.3,
}
TORCH_SEED = 8888

torch.manual_seed(TORCH_SEED)

alice_data, alice_targets = XORDataset(256).get()
bob_data, bob_targets, charlie_data, charlie_targets = XORDataset(
    128).split()
david_data, david_targets, eve_data, eve_targets = XORDataset(4).split()

    
def test_integration_crowdsource():
    """
    Integration test for crowdsource scenario.
    Alice is evaluator, others are trainers.
    """
    alice = CrowdsourceClient("Alice", alice_data, alice_targets, XORModel, F.mse_loss, 0)
    bob = CrowdsourceClient("Bob", bob_data, bob_targets, XORModel, F.mse_loss, 1)
    charlie = CrowdsourceClient("Charlie", charlie_data, charlie_targets, XORModel, F.mse_loss, 2)
    david = CrowdsourceClient("David", david_data, david_targets, XORModel, F.mse_loss, 3)
    eve = CrowdsourceClient("Eve", eve_data, eve_targets, XORModel, F.mse_loss, 4)

    print("Alice setting genesis...")
    alice.wait_for([
        alice.set_genesis_model(5)
    ])

    # Training
    for i in range(1, TRAINING_ITERATIONS+1):
        print(f"\nIteration {i}")
        alice.wait_for([
            bob.run_training_round(**TRAINING_HYPERPARAMS),
            charlie.run_training_round(**TRAINING_HYPERPARAMS),
            david.run_training_round(**TRAINING_HYPERPARAMS),
            eve.run_training_round(**TRAINING_HYPERPARAMS)
        ])
        print("\tAlice evaluating global...")
        print_global_performance(alice)
    
    # Retrospective evaluation
    pending_txs = []
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
    
    alice_global_model = alice.get_current_global_model()
    bob_global_model = bob.get_current_global_model()

    assert _same_weights(
        alice_global_model,
        bob_global_model
    ), "Alice and Bob ran the same aggregation but got different model weights"

    assert str(alice_global_model.state_dict()) == \
        str(bob_global_model.state_dict()), \
            "Alice and Bob ran the same aggregation but got different model dicts"

def test_integration_consortium():
    """
    Integration test for consortium setting.
    Alice sets up the main contract but doesn't participate.
    """
    alice = ConsortiumSetupClient("Alice", XORModel, 0)
    bob = ConsortiumClient("Bob", bob_data, bob_targets, XORModel, 1)
    charlie = ConsortiumClient("Charlie", charlie_data, charlie_targets, XORModel, 2)
    david = ConsortiumClient("David", david_data, david_targets, XORModel, 3)
    eve = ConsortiumClient("Eve", eve_data, eve_targets, XORModel, 4)

    alice.wait_for([
        alice.set_genesis_model(15)
    ])

    alice.wait_for([
        alice.add_sub(bob.address),
        alice.add_sub(charlie.address),
        alice.add_sub(david.address),
        alice.add_sub(eve.address),
    ])

    # Training
    for i in range(1, TRAINING_ITERATIONS+1):
        print(f"\nIteration {i}")
        print("\tTraining...")
        alice.wait_for([
            *bob.run_training_round(**TRAINING_HYPERPARAMS),
            *charlie.run_training_round(**TRAINING_HYPERPARAMS),
            *david.run_training_round(**TRAINING_HYPERPARAMS),
            *eve.run_training_round(**TRAINING_HYPERPARAMS)
        ])

    # Retrospective evaluation
    pending_txs = []
    for i in range(1, TRAINING_ITERATIONS+1):  # TODO: automatically find which rounds need evcaluating
        print(f"\nEvaluating iteration {i}")
        print("\tCalculating SVs...")
        bob_scores = bob.evaluate_updates(i)
        charlie_scores = charlie.evaluate_updates(i)
        david_scores = david.evaluate_updates(i)
        eve_scores = eve.evaluate_updates(i)
        print("\tSetting SVs...")
        pending_txs.extend([
            *bob.set_tokens(bob_scores),
            *charlie.set_tokens(charlie_scores),
            *david.set_tokens(david_scores),
            *eve.set_tokens(eve_scores)
        ])

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

    bob_global_model = bob.get_current_global_model()
    charlie_global_model = charlie.get_current_global_model()

    assert _same_weights(
        bob_global_model,
        charlie_global_model
    ), "Bob and Charlie ran the same aggregation but got different model weights"

    assert str(bob_global_model.state_dict()) == \
        str(charlie_global_model.state_dict()), \
            "Bob and Charlie ran the same aggregation but got different model dicts"



def _same_weights(model_a, model_b):
    """
    Checks if two pytorch models have the same weights.
    https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    """
    for params_a, params_b in zip(model_a.parameters(), model_b.parameters()):
        if (params_a.data!=params_b.data).sum() > 0:
            return False
    return True
