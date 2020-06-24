import os
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from clients import CrowdsourceClient
from utils import print_global_performance, print_token_count


class Data:
    def __init__(self, train):
        self._dataset = datasets.MNIST(
            'experiments/resources',
            train=train)
        self.data = torch.tensor(self._dataset.data
                                 ).float().view(-1, 1, 28, 28) / 255
        self.targets = torch.tensor(self._dataset.targets)

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


alice_data, alice_targets = Data(train=False).split(1)

bob_data, bob_targets, \
    charlie_data, charlie_targets, david_data, david_targets, \
    eve_data, eve_targets = Data(train=True).split(4)

# These clients will evaluate
alice = CrowdsourceClient("Alice", alice_data, alice_targets, Model, F.nll_loss, 0)

# These clients will train
bob = CrowdsourceClient("Bob", bob_data, bob_targets, Model, F.nll_loss, 1)
charlie = CrowdsourceClient("Charlie", charlie_data, charlie_targets, Model, F.nll_loss, 2)
david = CrowdsourceClient("David", david_data, david_targets, Model, F.nll_loss, 3)
eve = CrowdsourceClient("Eve", eve_data, eve_targets, Model, F.nll_loss, 4)

TRAINING_ITERATIONS = 3
TRAINING_HYPERPARAMETERS = {
    'epochs': 1,
    'learning_rate': 1e-2
}

tx = alice.set_genesis_model(30)
alice.wait_for_txs([tx])

for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nIteration {i}")

    print(f"\tBob training...")
    txb = bob._train_single_round(**TRAINING_HYPERPARAMETERS)
    print(f"\tCharlie training...")
    txc = charlie._train_single_round(**TRAINING_HYPERPARAMETERS)
    print(f"\tDavid training...")
    txd = david._train_single_round(**TRAINING_HYPERPARAMETERS)
    print(f"\tEve training...")
    txe = eve._train_single_round(**TRAINING_HYPERPARAMETERS)
    print(f"\tWaiting for transactions...")
    alice.wait_for_txs([txb, txc, txd, txe])
    print_global_performance(alice)

for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nEvaluating iteration {i}")
    scores = alice.evaluate_updates(i)
    txs = alice._set_tokens(scores)

alice.wait_for_txs(txs)
print_token_count(alice)
print_token_count(bob)
print_token_count(charlie)
print_token_count(david)
print_token_count(eve)
