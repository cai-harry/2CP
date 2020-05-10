import os
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from context import Client
from context import print_global_performance, print_token_count


class CovidData:
    def __init__(self):
        self._df = joblib.load(
            os.path.join(os.path.dirname(__file__),
                         "resources", "covid_data.bin")
        )
        X, y = self._split_into_lists(
            data=self._df.drop(['ICU'], 1),
            labels=self._df['ICU']
        )
        self.data = self._convert_to_tensor(X)
        self.targets = self._convert_to_tensor(y).unsqueeze(dim=-1)

    def split_3(self):
        left = len(self.data) // 3
        right = 2 * len(self.data) // 3
        return (
            self.data[:left],
            self.targets[:left],
            self.data[left:right],
            self.targets[left:right],
            self.data[right:],
            self.targets[right:]
        )

    def _split_into_lists(self, data, labels):
        record_list = list()
        result_list = list()
        for _, row in data.iterrows():
            record_list.append(row)
        for row in labels:
            converted_label = float(row)
            result_list.append(converted_label)
        return record_list, result_list

    def _convert_to_tensor(self, data):
        tensors = []
        for record in data:
            tensors.append(torch.tensor(record))
        return torch.stack(tensors)


class CovidModel(nn.Module):
    def __init__(self):
        super(CovidModel, self).__init__()
        self.fc1 = nn.Linear(62, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x



alice_data, alice_targets, bob_data, bob_targets, \
     charlie_data, charlie_targets = CovidData().split_3()

print(alice_data.shape)
print(alice_targets.shape)

contract_address = input("Enter contract address:\n>")

# These clients will evaluate
alice = Client("Alice", alice_data, alice_targets, CovidModel, contract_address, 0)

# These clients will train
bob = Client("Bob", bob_data, bob_targets, CovidModel, contract_address, 1)
charlie = Client("Charlie", charlie_data, charlie_targets, CovidModel, contract_address, 2)

TRAINING_ITERATIONS = 16
LEARNING_RATE = 1e-2

alice.set_genesis_model()
for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nIteration {i}")
    bob.run_training_round(LEARNING_RATE)
    charlie.run_training_round(LEARNING_RATE)
    print_global_performance(alice)
    alice.finish_training_round()
    print_token_count(bob)
    print_token_count(charlie)
