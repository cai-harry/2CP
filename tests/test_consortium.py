import threading

import torch
import torch.nn.functional as F

from clients import ConsortiumSetupClient, ConsortiumClient
from utils import print_token_count

from test_utils.xor import XORDataset, XORModel
from test_utils.functions import same_weights

TRAINING_ITERATIONS = 2
TRAINING_HYPERPARAMS = {
    'final_round_num': TRAINING_ITERATIONS,
    'batch_size': 64,
    'epochs': 2,
    'learning_rate': 0.3,
}
EVAL_METHOD = 'step'
TORCH_SEED = 8888

ROUND_DURATION = 60  # expecting rounds to always end early

torch.manual_seed(TORCH_SEED)

bob_data, bob_targets, charlie_data, charlie_targets = XORDataset(
    64).split()
david_data, david_targets, eve_data, eve_targets = XORDataset(4).split()


def test_consortium():
    """
    Integration test for consortium setting.
    Alice sets up the main contract but doesn't participate.
    """
    alice = ConsortiumSetupClient("Alice", XORModel, 0, deploy=True)
    bob = ConsortiumClient(
        "Bob", bob_data, bob_targets, XORModel, F.mse_loss, 1,
        contract_address=alice.contract_address)
    charlie = ConsortiumClient(
        "Charlie", charlie_data, charlie_targets, XORModel, F.mse_loss, 2,
        contract_address=alice.contract_address)
    david = ConsortiumClient(
        "David", david_data, david_targets, XORModel, F.mse_loss, 3,
        contract_address=alice.contract_address)

    trainers = [bob,
                charlie,
                david]

    alice.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=len(trainers)
    )

    alice.add_auxiliaries([
        trainer.address for trainer in trainers
    ])

    # Training
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS,
            daemon=True
        ) for trainer in trainers
    ]

    # Evaluation
    threads.extend([
        threading.Thread(
            target=trainer.evaluate_until,
            args=(TRAINING_ITERATIONS, EVAL_METHOD),
            daemon=True
        ) for trainer in trainers
    ])

    # Run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print_token_count(bob)
    print_token_count(charlie)
    print_token_count(david)

    assert bob.get_token_count()[0] > 0, "Bob ended up with 0 tokens"
    assert charlie.get_token_count()[0] > 0, "Charlie ended up with 0 tokens"

    bob_global_model = bob.get_current_global_model()
    charlie_global_model = charlie.get_current_global_model()

    assert same_weights(
        bob_global_model,
        charlie_global_model
    ), "Bob and Charlie ran the same aggregation but got different model weights"

    assert str(bob_global_model.state_dict()) == \
        str(charlie_global_model.state_dict()), \
        "Bob and Charlie ran the same aggregation but got different model dicts"
