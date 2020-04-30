import os
import json

import base58
import ipfshttpclient
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from web3 import Web3, HTTPProvider


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x.squeeze()


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

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    CONTRACT_JSON_PATH = "build/contracts/FederatedLearning.json"
    IPFS_HASH_PREFIX = bytes.fromhex('1220')

    def __init__(self, contract_address, account_idx):
        self._web3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))
        self._contract = self._instantiate_contract(contract_address)
        self._web3.eth.defaultAccount = self._web3.eth.accounts[account_idx]

    def getPreviousUpdates(self):
        list_hash_bytes = self._contract.functions.getPreviousUpdates().call()
        return [self._from_bytes32(hash_bytes) for hash_bytes in list_hash_bytes]

    def setGenesis(self, model_hash):
        bytes_to_store = self._to_bytes32(model_hash)
        self._contract.functions.setGenesis(bytes_to_store).transact()

    def addTrainer(self):
        self._contract.functions.addTrainer().transact()

    def addModelUpdate(self, model_hash):
        bytes_to_store = self._to_bytes32(model_hash)
        self._contract.functions.addModelUpdate(bytes_to_store).transact()

    def _instantiate_contract(self, contract_address):
        with open(self.CONTRACT_JSON_PATH) as crt_json:
            abi = json.load(crt_json)['abi']
        instance = self._web3.eth.contract(
            abi=abi,
            address=contract_address
        )
        return instance

    def _to_bytes32(self, model_hash):
        bytes34 = base58.b58decode(model_hash)
        assert bytes34[:2] == self.IPFS_HASH_PREFIX, \
            f"IPFS hash should begin with {self.IPFS_HASH_PREFIX} but got {bytes34[:2].hex()}"
        bytes32 = bytes34[2:]
        return bytes32

    def _from_bytes32(self, bytes32):
        bytes34 = self.IPFS_HASH_PREFIX + bytes32
        model_hash = base58.b58encode(bytes34).decode()
        return model_hash


class Client:
    def __init__(self, name, data, contract_address, account_idx):
        self._name = name
        self._data = data
        self._data_loader = torch.utils.data.DataLoader(
            self._data,
            batch_size=len(self._data),
            shuffle=True
        )
        self._contract = ContractClient(contract_address, account_idx)
        self._ipfs_client = IPFSClient()
        self._registered_trainer = False

    def set_genesis_model(self):
        genesis_model = Model()
        genesis_hash = self._ipfs_client.add_model(genesis_model)
        self._contract.setGenesis(genesis_hash)

    def run_training_round(self):
        if not self._registered_trainer:
            self._register_as_trainer()
        model = self._get_global_model()
        model = self._train_model(model)
        uploaded_hash = self._upload_model(model)
        self._record_model(uploaded_hash)

    def evaluate(self):
        model = self._get_global_model()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)
                total_loss += F.mse_loss(pred, labels)
                total_correct += (torch.round(pred) == labels).float().sum()
        avg_loss = total_loss / len(self._data)
        accuracy = total_correct / len(self._data)
        return avg_loss.item(), accuracy

    def predict_and_plot(self):
        model = self._get_global_model()
        with torch.no_grad():
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)
                plt.scatter(
                    data[:, 0], data[:, 1], c=pred,
                    cmap='bwr')
                plt.scatter(
                data[:, 0], data[:, 1], c=torch.round(pred),
                cmap='bwr', marker='+')
        plt.show()

    def _register_as_trainer(self):
        self._contract.addTrainer()
        self._registered_trainer = True

    def _get_global_model(self):
        """
        Calculate current global model by aggregating all updates from previous round.
        """
        model_hashes = self._get_model_hashes()
        models = self._get_models(model_hashes)
        avg_model = self._avg_model(models)
        return avg_model

    def _train_model(self, model):
        optimizer = optim.SGD(model.parameters(), lr=0.3)
        for data, labels in self._data_loader:
            data, labels = data.float(), labels.float()
            optimizer.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, labels)
            loss.backward()
            optimizer.step()
        return model

    def _upload_model(self, model):
        """Uploads the given model to IPFS."""
        uploaded_hash = self._ipfs_client.add_model(model)
        return uploaded_hash

    def _record_model(self, uploaded_hash):
        """Records the given model IPFS hash on the smart contract."""
        self._contract.addModelUpdate(uploaded_hash)

    def _get_model_hashes(self):
        return self._contract.getPreviousUpdates()

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
