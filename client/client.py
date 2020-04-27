import os

import base58
import ipfshttpclient
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
import web3

import mock_contract


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.squeeze()
        return x


class IPFSClient:
    def __init__(self):
        self._ipfs_client = ipfshttpclient.connect(
            '/ip4/127.0.0.1/tcp/5001/http')

    def get_model(self, model_hash):
        model = Model()
        with self._ipfs_client as ipfs:
            # saves to current directory, filename is model_hash
            ipfs.get(model_hash)
        model.load_state_dict(torch.load(model_hash))
        os.remove(model_hash)
        return model

    def add_model(self, model):
        model_filename = "tmp.pt"
        with self._ipfs_client as ipfs:
            torch.save(model.state_dict(), model_filename)
            res = ipfs.add(model_filename)
        os.remove(model_filename)
        model_hash = res['Hash']
        return model_hash


class ContractClient:
    def __init__(self, contract):
        self._contract = contract

    def recordContribution(self, model_hash):
        self._contract.recordContribution(
            self._to_bytes32(model_hash)
        ).transact()

    def getContributions(self):
        res = self._contract.getContributions.call()
        return self._from_bytes32(res)

    def getLatestHash(self):
        res = self._contract.getLatestHash.call()
        return self._from_bytes32(res)

    def setLatestHash(self, model_hash):
        """Sets new global model and resets list of updates"""
        self.latestHash = model_hash
        self.updates = []
        print(model_hash)

    def _to_bytes32(self, model_hash):
        full_hex = hex(base58.b58decode_int(model_hash))
        assert full_hex[:6] == "0x1220", "Model hash should begin with 0x1220"
        return full_hex[:6]  # remove beginning "0x1220"

    def _from_bytes32(self, bytes32):
        full_hex = "0x1220"+bytes32
        full_int = int(full_hex, 16)
        return base58.b58encode_int(full_int)


class Client:
    def __init__(self, data, batch_size, contract):
        self._data = data
        self._batch_size = batch_size
        self._contract = contract
        self._ipfs_client = IPFSClient()
        self._data_iterator = self._reset_data_iterator()

    def run_train(self):
        model = self._get_latest_model()
        model = self._train_model(model)
        uploaded_hash = self._upload_model(model)
        self._record_contribution(uploaded_hash)

    def _get_latest_model(self):
        latest_hash = self._contract.getLatestHash()
        model = self._ipfs_client.get_model(latest_hash)
        return model

    def _train_model(self, model):
        data, label = self._get_next_data_and_label()
        optimizer = optim.SGD(model.parameters(), lr=0.3)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        return model

    def _upload_model(self, model):
        uploaded_hash = self._ipfs_client.add_model(model)
        return uploaded_hash

    def _record_contribution(self, uploaded_hash):
        self._contract.recordContribution(uploaded_hash)

    def _reset_data_iterator(self):
        return iter(torch.utils.data.DataLoader(
            self._data,
            batch_size=self._batch_size,
            shuffle=True
        ))

    def _get_next_data_and_label(self):
        try:
            data, labels = next(self._data_iterator)
        except StopIteration:
            # Client has no more data, so reset iterator
            self._data_iterator = self._reset_data_iterator()
            data, labels = next(self._data_iterator)
        data = data.float()
        labels = labels.float()
        return data, labels


class Org:
    def __init__(self, data, contract):
        self._ipfs_client = IPFSClient()
        self._contract = contract
        self._latest_model = None
        self._data = data
        self._data_loader = torch.utils.data.DataLoader(
            self._data,
            batch_size=len(self._data)
        )
        self._set_genesis_model()

    def run_aggregation(self):
        model_hashes = self._get_model_hashes()
        models = self._get_models(model_hashes)
        avg_model = self._avg_model(models)
        self._set_latest_model(avg_model)
        self._latest_model = avg_model

    def evaluate(self):
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = self._latest_model(data)
                total_loss += F.mse_loss(pred, labels)
                total_correct += (torch.round(pred) == labels).float().sum()
        avg_loss = total_loss / len(self._data)
        accuracy = total_correct / len(self._data)
        return avg_loss.item(), accuracy

    def predict_and_plot(self):
        with torch.no_grad():
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = self._latest_model(data)
                plt.scatter(
                    data[:, 0], data[:, 1], c=pred, 
                    cmap='bwr')
                plt.scatter(
                    data[:, 0], data[:, 1], c=torch.round(pred),
                    cmap='bwr', marker='+')
                plt.show()

    def _set_genesis_model(self):
        genesis_model = Model()
        genesis_hash = self._ipfs_client.add_model(genesis_model)
        self._contract.setLatestHash(genesis_hash)
        self._latest_model = genesis_model

    def _get_model_hashes(self):
        return self._contract.getContributions()

    def _get_models(self, model_hashes):
        models = []
        for model_hash in model_hashes:
            models.append(self._ipfs_client.get_model(model_hash))
        return models

    def _avg_model(self, models):
        avg_model = Model()
        with torch.no_grad():
            for params in avg_model.parameters():
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model

    def _set_latest_model(self, avg_model):
        latest_hash = self._ipfs_client.add_model(avg_model)
        self._contract.setLatestHash(latest_hash)
