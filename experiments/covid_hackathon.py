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
        self._data = self._convert_to_tensors(X)
        self._labels = y

    def split_3(self):
        left = len(self._data) // 3
        right = 2 * len(self._data) // 3
        data_0 = self._data[:left]
        labels_0 = self._labels[:left]
        dataset_0 = list(zip(data_0, labels_0))
        data_1 = self._data[left:right]
        labels_1 = self._labels[left:right]
        dataset_1 = list(zip(data_1, labels_1))
        data_2 = self._data[right:]
        labels_2 = self._labels[right:]
        dataset_2 = list(zip(data_2, labels_2))
        return dataset_0, dataset_1, dataset_2

    def _split_into_lists(self, data, labels):
        record_list = list()
        result_list = list()
        for _, row in data.iterrows():
            record_list.append(row)
        for row in labels:
            converted_label = float(row)
            result_list.append(converted_label)
        return record_list, result_list

    def _convert_to_tensors(self, data):
        tensors = list()
        for record in data:
            tensors.append(torch.tensor(record))
        return tensors


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
        return x.squeeze()


contract_address = input("Enter contract address:\n>")

alice_data, bob_data, charlie_data = CovidData().split_3()

# These clients will evaluate
alice = Client("Alice", alice_data, CovidModel, contract_address, 0)

# These clients will train
bob = Client("Bob", bob_data, CovidModel, contract_address, 1)
charlie = Client("Charlie", charlie_data, CovidModel, contract_address, 2)

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
