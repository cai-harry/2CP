import os

import ipfshttpclient
import torch
from torch import nn, optim
from torch.nn import functional as F
import web3

import mock_contract

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class IPFSClient:
    def __init__(self):
        self._ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
    
    def get_model(self, model_hash):
        model = Model()
        with self._ipfs_client as ipfs:
            ipfs.get(model_hash)  # saves to current directory, filename is model_hash
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

class Client:
    def __init__(self, data, contract):
        self._data = data
        self._data_idx = 0
        self._ipfs_client = IPFSClient()
        self._contract = contract
        
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
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        pred = model(data).flatten()
        loss = F.mse_loss(pred, label)
        loss.backward()
        optimizer.step()
        return model
    
    def _upload_model(self, model):
        uploaded_hash = self._ipfs_client.add_model(model)
        return uploaded_hash
    
    def _record_contribution(self, uploaded_hash):
        self._contract.recordContribution(uploaded_hash)

    def _get_next_data_and_label(self):
        data, label = self._data[self._data_idx]
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()
        self._data_idx = (self._data_idx + 1) % len(self._data)
        return data, label
            
        
class Org:
    def __init__(self, contract):
        self._ipfs_client = IPFSClient()
        self._contract = contract
        self._set_genesis_model()
    
    def run_aggregation(self):
        model_hashes = self._get_model_hashes()
        models = self._get_models(model_hashes)
        avg_model = self._avg_model(models)
        self._set_latest_model(avg_model)

    def _set_genesis_model(self):
        genesis_model = Model()
        genesis_hash = self._ipfs_client.add_model(genesis_model)
        self._contract.setLatestHash(genesis_hash)
        
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