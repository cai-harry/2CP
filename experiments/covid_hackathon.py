import os
import joblib
import matplotlib.pyplot as plt
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from clients import CrowdsourceClient
from utils import print_global_performance, print_token_count


class CovidData:
    def __init__(self):
        self._df = joblib.load(
            os.path.join(os.path.dirname(__file__),
                        "covid_data.bin")
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
        super().__init__()
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
alice = CrowdsourceClient(
    "Alice", 
    alice_data, 
    alice_targets, 
    CovidModel, 
    F.mse_loss, 
    0,
    None,
    True
)

# These clients will train
bob = CrowdsourceClient(
    "Bob", 
    bob_data, 
    bob_targets, 
    CovidModel, 
    F.mse_loss, 
    1, 
    alice.contract_address, 
    False
)
charlie = CrowdsourceClient(
    "Charlie", 
    charlie_data, 
    charlie_targets, 
    CovidModel, 
    F.mse_loss, 
    2, 
    alice.contract_address, 
    False
)
david = CrowdsourceClient(
    "David", 
    david_data, 
    david_targets, 
    CovidModel, 
    F.mse_loss, 
    3, 
    alice.contract_address, 
    False
)
eve = CrowdsourceClient(
    "Eve", 
    eve_data, 
    eve_targets, 
    CovidModel, 
    F.mse_loss, 
    4, 
    alice.contract_address, 
    False
)
trainers = [bob, charlie, david, eve]

TRAINING_ITERATIONS = 16
TRAINING_HYPERPARAMETERS = {
    'final_round_num': TRAINING_ITERATIONS,
    'batch_size': 8,
    'epochs': 10,
    'learning_rate': 1e-2
}

alice.set_genesis_model(
    round_duration=120,
    max_num_updates=4
)

threads = [
    threading.Thread(
        target=trainer.train_until,
        kwargs=TRAINING_HYPERPARAMETERS,
        daemon=True
    ) for trainer in trainers
]

# define evaluation threads
threads.append(
    threading.Thread(
        target=alice.evaluate_until,
        args=(TRAINING_ITERATIONS, 'step'),
        daemon=True
    )
)

# run all threads in parallel
for t in threads:
    t.start()
for t in threads:
    t.join()

print_token_count(alice)
print_token_count(bob)
print_token_count(charlie)
print_token_count(david)
print_token_count(eve)
