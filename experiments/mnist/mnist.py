import json
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from clients import CrowdsourceClient
from utils import print_global_performance, print_token_count


class Data:
    def __init__(self, train):
        self._dataset = datasets.MNIST(
            'experiments/mnist/resources',
            train=train)
        self.data = self._dataset.data.float().view(-1, 1, 28, 28) / 255
        self.targets = self._dataset.targets

    def split(self, n):
        perm = torch.randperm(len(self.data))
        chunks = torch.chunk(perm, n)
        output = []
        for c in chunks:
            output.append(self.data[c])
            output.append(self.targets[c])
        return output


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


QUICK_RUN = False

TRAINING_ITERATIONS = 3
TRAINING_HYPERPARAMS = {
    'final_round_num': TRAINING_ITERATIONS,
    'batch_size': 32,
    'epochs': 1,
    'learning_rate': 1e-2
}
ROUND_DURATION = 1000  # should always end early


def run_crowdsource_experiment(
        num_trainers,
        seed):
    """
    Experiment 1: fairness / variance
    iid datasets; expecting clients to get similar scores. Varying number of trainers
    """
    if not 2 <= num_trainers <= 6:
        raise ValueError(
            f"Expected num_trainers to be between 2 and 6, got {num_trainers}")

    # set up results dict, add details of current experiment
    results = {}
    results['num_trainers'] = num_trainers
    results['seed'] = seed

    torch.manual_seed(seed)

    # instantiate data
    if QUICK_RUN:
        alice_data, alice_targets, *_ = Data(train=False).split(100)
    else:
        alice_data, alice_targets = Data(train=False).split(1)

    if QUICK_RUN:
        bob_data, bob_targets, \
            carol_data, carol_targets, \
            david_data, david_targets, \
            eve_data, eve_targets, \
            frank_data, frank_targets, \
            georgia_data, georgia_targets, \
            *_ = Data(train=True).split(60)
    else:
        bob_data, bob_targets, \
            carol_data, carol_targets, \
            david_data, david_targets, \
            eve_data, eve_targets, \
            frank_data, frank_targets, \
            georgia_data, georgia_targets \
            = Data(train=True).split(6)

    # instantiate clients
    alice = CrowdsourceClient(
        name="Alice",
        data=alice_data,
        targets=alice_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=0,
        contract_address=None,
        deploy=True
    )
    bob = CrowdsourceClient(
        name="Bob",
        data=bob_data,
        targets=bob_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=1,
        contract_address=alice.contract_address,
        deploy=False
    )
    carol = CrowdsourceClient(
        name="Carol",
        data=carol_data,
        targets=carol_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=2,
        contract_address=alice.contract_address,
        deploy=False
    )
    david = CrowdsourceClient(
        name="David",
        data=david_data,
        targets=david_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=3,
        contract_address=alice.contract_address,
        deploy=False
    )
    eve = CrowdsourceClient(
        name="Eve",
        data=eve_data,
        targets=eve_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=4,
        contract_address=alice.contract_address,
        deploy=False
    )
    frank = CrowdsourceClient(
        name="Frank",
        data=frank_data,
        targets=frank_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=5,
        contract_address=alice.contract_address,
        deploy=False
    )
    georgia = CrowdsourceClient(
        name="Georgia",
        data=georgia_data,
        targets=georgia_targets,
        model_constructor=Model,
        model_criterion=F.nll_loss,
        account_idx=6,
        contract_address=alice.contract_address,
        deploy=False
    )
    results['contract_address'] = alice.contract_address

    trainers = [
        bob,
        carol,
        david,
        eve,
        frank,
        georgia
    ][:num_trainers]

    # Set up
    alice.set_genesis_model(
        round_duration=ROUND_DURATION,
        max_num_updates=len(trainers)
    )

    # define training threads
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS,
            daemon=True
        ) for trainer in trainers
    ]

    # define evaluation threads
    threads.append(
        threading.Thread(
            target=alice.evaluate_until,
            args=(TRAINING_ITERATIONS,),
            daemon=True
        )
    )

    # run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results['final_tokens'] = [trainer.get_token_count()[0]
                               for trainer in trainers]

    filename = f"experiments/mnist/results/crowdsource-{num_trainers}-{seed}.json"
    with open(filename, 'w') as f:
        json.dump(results, f,
                  indent=4)
    print(f"Saved to {filename}")

    return results


if __name__ == "__main__":
    experiments = [
        # {'num_trainers': 2, 'seed': 32},
        # {'num_trainers': 3, 'seed': 32},
        # {'num_trainers': 4, 'seed': 32},
        # {'num_trainers': 5, 'seed': 32},
        # {'num_trainers': 6, 'seed': 32},
        # {'num_trainers': 2, 'seed': 76},
        # {'num_trainers': 3, 'seed': 76},
        # {'num_trainers': 4, 'seed': 76},
        # {'num_trainers': 5, 'seed': 76},
        # {'num_trainers': 6, 'seed': 76},
        {'num_trainers': 2, 'seed': 88},
        {'num_trainers': 3, 'seed': 88},
        {'num_trainers': 4, 'seed': 88},
        {'num_trainers': 5, 'seed': 88}
        # {'num_trainers': 6, 'seed': 88}
    ]
    if QUICK_RUN:
        experiments = experiments[:2]
    for exp in experiments:
        run_crowdsource_experiment(**exp)
