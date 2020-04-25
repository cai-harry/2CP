import matplotlib.pyplot as plt
import numpy as np
import torch

class XOR:
    """
    A dataset of uniformly scattered 2d points.
    Points with xy >= 0 are class A
    Points with xy < 0 are class B
    """
    def __init__(self, num_pts):
        self._data = np.random.uniform(-1, 1, size=(num_pts, 2))
        self._labels = (self._data[:, 0] * self._data[:, 1] < 0).astype(int)

    def split_by_label(self):
        data_0 = self._data[self._labels==0]
        labels_0 = self._labels[self._labels==0]
        dataset_0 = list(zip(data_0, labels_0))
        data_1 = self._data[self._labels==1]
        labels_1 = self._labels[self._labels==1]
        dataset_1 = list(zip(data_1, labels_1))
        return dataset_0, dataset_1

def test_XOR():
    alice_data, bob_data = XOR(100).split_by_label()
    plt.scatter(alice_data[0][:,0], alice_data[0][:,1], label='Alice')
    plt.scatter(bob_data[0][:,0], bob_data[0][:,1], label='Bob')
    plt.legend()
    plt.show()