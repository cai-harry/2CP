import threading

import torch
import torch.nn.functional as F

from clients import CrowdsourceClient
from utils import print_token_count

from test_utils.xor import XORDataset, XORModel
from test_utils.functions import same_weights

TRAINING_ITERATIONS = 2
TRAINING_HYPERPARAMS = {
    'final_round_num': TRAINING_ITERATIONS,
    'epochs': 2,
    'learning_rate': 0.3,
}
TORCH_SEED = 8888

ROUND_DURATION = 10

torch.manual_seed(TORCH_SEED)

alice_data, alice_targets = XORDataset(128).get()
bob_data, bob_targets, charlie_data, charlie_targets = XORDataset(
    64).split()
david_data, david_targets, eve_data, eve_targets = XORDataset(4).split()


def test_crowdsource():
    """
    Integration test for crowdsource scenario.
    Alice is evaluator, others are trainers.
    """
    alice = CrowdsourceClient(
        "Alice", alice_data, alice_targets, XORModel, F.mse_loss, 0)
    bob = CrowdsourceClient("Bob", bob_data, bob_targets,
                            XORModel, F.mse_loss, 1)
    charlie = CrowdsourceClient(
        "Charlie", charlie_data, charlie_targets, XORModel, F.mse_loss, 2)
    david = CrowdsourceClient(
        "David", david_data, david_targets, XORModel, F.mse_loss, 3)
    eve = CrowdsourceClient("Eve", eve_data, eve_targets,
                            XORModel, F.mse_loss, 4)

    # alice is evaluator
    # others are trainers
    trainers = [bob, charlie, david, eve]

    alice.set_genesis_model(ROUND_DURATION)

    # Training
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS
        ) for trainer in trainers
    ]

    # Evaluation
    threads.append(
        threading.Thread(
            target=alice.evaluate_until,
            args=(TRAINING_ITERATIONS,)
        )
    )

    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print_token_count(bob)
    print_token_count(charlie)
    print_token_count(david)
    print_token_count(eve)

    assert bob.get_token_count()[0] > david.get_token_count(
    )[0], "Bob ended up with fewer tokens than David"
    assert bob.get_token_count()[0] > eve.get_token_count(
    )[0], "Bob ended up with fewer tokens than Eve"
    assert charlie.get_token_count()[0] > david.get_token_count(
    )[0], "Charlie ended up with fewer tokens than David"
    assert charlie.get_token_count()[0] > eve.get_token_count(
    )[0], "Charlie ended up with fewer tokens than Eve"

    alice_global_model = alice.get_current_global_model()
    bob_global_model = bob.get_current_global_model()

    assert same_weights(
        alice_global_model,
        bob_global_model
    ), "Alice and Bob ran the same aggregation but got different model weights"

    assert str(alice_global_model.state_dict()) == \
        str(bob_global_model.state_dict()), \
        "Alice and Bob ran the same aggregation but got different model dicts"
