import os
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from client import CrowdsourceClient
from utils import print_global_performance, print_token_count


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
        assert len(self.data) == len(self.targets), \
            f"Data and targets have different lengths {len(self.data)} and {len(self.targets)}"

    def split(self, n):
        perm = torch.randperm(len(self.data))
        chunks = torch.chunk(perm, n)
        output = []
        for c in chunks:
            output.append(self.data[c])
            output.append(self.targets[c])
        return output


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
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x



alice_data, alice_targets, bob_data, bob_targets, \
     charlie_data, charlie_targets, david_data, david_targets, \
         eve_data, eve_targets = CovidData().split(5)

# These clients will evaluate
alice = CrowdsourceClient("Alice", alice_data, alice_targets, CovidModel, 0)

# These clients will train
bob = CrowdsourceClient("Bob", bob_data, bob_targets, CovidModel, 1)
charlie = CrowdsourceClient("Charlie", charlie_data, charlie_targets, CovidModel, 2)
david = CrowdsourceClient("David", david_data, david_targets, CovidModel, 3)
eve = CrowdsourceClient("Eve", eve_data, eve_targets, CovidModel, 4)

TRAINING_ITERATIONS = 16
TRAINING_HYPERPARAMETERS = {
    'epochs': 64,
    'learning_rate': 1e-2
}

tx = alice.set_genesis_model()
alice.wait_for([tx])

for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nIteration {i}")
    
    txb = bob.run_training_round(**TRAINING_HYPERPARAMETERS)
    txc = charlie.run_training_round(**TRAINING_HYPERPARAMETERS)
    txd = david.run_training_round(**TRAINING_HYPERPARAMETERS)
    txe = eve.run_training_round(**TRAINING_HYPERPARAMETERS)
    alice.wait_for([txb, txc, txd, txe])
    print_global_performance(alice)

for i in range(1, TRAINING_ITERATIONS+1):
    print(f"\nEvaluating iteration {i}")
    scores = alice.evaluate_updates(i)
    txs = alice.set_tokens(scores)

alice.wait_for(txs)
print_token_count(alice)
print_token_count(bob)
print_token_count(charlie)
print_token_count(david)
print_token_count(eve)
