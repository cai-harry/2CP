import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class XORData:
    """
    A dataset of uniformly scattered 2d points.
    Points with xy >= 0 are class A
    Points with xy < 0 are class B
    """

    def __init__(self, num_pts):
        self._data = np.random.uniform(-1, 1, size=(num_pts, 2))
        self._labels = (self._data[:, 0] * self._data[:, 1] < 0).astype(float)

    def as_dataset(self):
        return list(zip(self._data, self._labels))

    def split(self):
        mid = len(self._data) // 2
        data_0 = self._data[:mid]
        labels_0 = self._labels[:mid]
        dataset_0 = list(zip(data_0, labels_0))
        data_1 = self._data[mid:]
        labels_1 = self._labels[mid:]
        dataset_1 = list(zip(data_1, labels_1))
        return dataset_0, dataset_1

    def split_by_label(self):
        data_0 = self._data[self._labels == 0]
        labels_0 = self._labels[self._labels == 0]
        dataset_0 = list(zip(data_0, labels_0))
        data_1 = self._data[self._labels == 1]
        labels_1 = self._labels[self._labels == 1]
        dataset_1 = list(zip(data_1, labels_1))
        return dataset_0, dataset_1


def test_XOR():
    alice_data, bob_data = XORData(100).split_by_label()
    plt.scatter(alice_data[0][:, 0], alice_data[0][:, 1], label='Alice')
    plt.scatter(bob_data[0][:, 0], bob_data[0][:, 1], label='Bob')
    plt.legend()
    plt.show()


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x.squeeze()


def plot_predictions(data, pred):
    plt.scatter(
        data[:, 0], data[:, 1], c=pred,
        cmap='bwr')
    plt.scatter(
        data[:, 0], data[:, 1], c=torch.round(pred),
        cmap='bwr', marker='+')
    plt.show()
