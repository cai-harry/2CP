import os
import json

import base58
import ipfshttpclient
import matplotlib.pyplot as plt
import syft as sy
import torch
from torch import nn, optim
from torch.nn import functional as F
from web3 import Web3, HTTPProvider

import shapley


class IPFSClient:
    def __init__(self):
        self._ipfs_client = ipfshttpclient.connect(
            '/ip4/127.0.0.1/tcp/5001/http')

    def get_model(self, model_hash, model_constructor):
        model = model_constructor()
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
    """
    Wrapper over the Smart Contract ABI, to gracefully bridge Python data to Solidity.

    The API of this class should match that of the smart contract.
    """

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    CONTRACT_JSON_PATH = "build/contracts/FederatedLearning.json"
    IPFS_HASH_PREFIX = bytes.fromhex('1220')

    def __init__(self, contract_address, account_idx):
        self._web3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))
        self._contract = self._instantiate_contract(contract_address)
        self._address = self._web3.eth.accounts[account_idx]

        self._web3.eth.defaultAccount = self._address

    def evaluator(self):
        return self._contract.functions.evaluator().call()

    def trainer(self, idx):
        return self._contract.functions.trainers(idx).call()

    def genesis(self):
        hash_bytes = self._contract.functions.genesis().call()
        return self._from_bytes32(hash_bytes)

    def previousUpdates(self, trainer_address):
        hash_bytes = self._contract.functions.previousUpdates(
            trainer_address).call()
        return self._from_bytes32(hash_bytes)

    def currentUpdates(self, trainer_address):
        hash_bytes = self._contract.functions.currentUpdates(
            trainer_address).call()
        return self._from_bytes32(hash_bytes)

    def trainingRound(self):
        return self._contract.functions.trainingRound().call()

    def tokens(self, trainer_address):
        return self._contract.functions.tokens(trainer_address).call()

    def getTokens(self):
        return self._contract.functions.getTokens().call()

    def totalTokens(self):
        return self._contract.functions.totalTokens().call()

    def getTrainers(self):
        return self._contract.functions.getTrainers().call()

    def isTrainer(self):
        return self._contract.functions.isTrainer().call()

    def isTrainingRoundFinished(self):
        return self._contract.functions.isTrainingRoundFinished().call()

    def setGenesis(self, model_hash):
        bytes_to_store = self._to_bytes32(model_hash)
        self._contract.functions.setGenesis(bytes_to_store).transact()

    def addTrainer(self):
        self._contract.functions.addTrainer().transact()

    def addModelUpdate(self, model_hash):
        bytes_to_store = self._to_bytes32(model_hash)
        self._contract.functions.addModelUpdate(bytes_to_store).transact()

    def giveTokens(self, trainer_address, num_tokens):
        self._contract.functions.giveTokens(
            trainer_address, num_tokens).transact()

    def nextTrainingRound(self):
        self._contract.functions.nextTrainingRound().transact()

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

    TOKENS_PER_UNIT_LOSS = 1e18 # number of wei per ether

    def __init__(self, name, data, model_constructor, contract_address, account_idx):
        self.name = name
        
        self._data = data
        self._data_loader = torch.utils.data.DataLoader(
            self._data,
            batch_size=len(self._data),
            shuffle=True
        )
        self._model_constructor = model_constructor
        self._contract = ContractClient(contract_address, account_idx)
        self._ipfs_client = IPFSClient()

    def set_genesis_model(self):
        genesis_model = self._model_constructor()
        genesis_hash = self._ipfs_client.add_model(genesis_model)
        self._contract.setGenesis(genesis_hash)

    def run_training_round(self, learning_rate):
        if not self._contract.isTrainer():
            self._contract.addTrainer()
        model = self._get_global_model()
        model = self._train_model(model, learning_rate)
        uploaded_hash = self._upload_model(model)
        self._record_model(uploaded_hash)

    def evaluate_global(self):
        model = self._get_global_model()
        return self._evaluate_model(model)

    def evaluate_trainers(self):
        """
        Provide Shaply Value score for each trainer in the training round.
        """
        trainers = self._contract.getTrainers()
        return shapley.values(
                self._characteristic_function, trainers)

    def finish_training_round(self):
        """
        Wraps up the current training round and starts a new one

        Evaluates contributions in the current round and distributes tokens
        """
        scores = self.evaluate_trainers()
        for trainer, score in scores.items():
            num_tokens = max(0, int(self.TOKENS_PER_UNIT_LOSS * score))
            self._contract.giveTokens(trainer, num_tokens)
        self._contract.nextTrainingRound()

    def predict(self):
        model = self._get_global_model()
        predictions = []
        with torch.no_grad():
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)
                predictions.append(pred)
        return torch.stack(predictions)

    def get_token_count(self):
        return self._contract.getTokens(), self._contract.totalTokens()

    def _get_global_model(self):
        """
        Calculate current global model by aggregating all updates from previous round.
        """
        if self._contract.trainingRound() == 0:
            return self._get_genesis_model()
        model_hashes = self._get_previous_update_hashes()
        models = self._get_models(model_hashes).values()
        avg_model = self._avg_model(models)
        return avg_model

    def _train_model(self, model, lr):
        optimizer = optim.SGD(model.parameters(), lr=lr)
        for data, labels in self._data_loader:
            data, labels = data.float(), labels.float()
            optimizer.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, labels)
            loss.backward()
            optimizer.step()
        return model

    def _evaluate_model(self, model):
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
        return avg_loss.item(), accuracy.item()

    def _upload_model(self, model):
        """Uploads the given model to IPFS."""
        uploaded_hash = self._ipfs_client.add_model(model)
        return uploaded_hash

    def _record_model(self, uploaded_hash):
        """Records the given model IPFS hash on the smart contract."""
        self._contract.addModelUpdate(uploaded_hash)

    def _get_previous_update_hashes(self, trainers=None):
        if trainers is None:
            trainers = self._contract.getTrainers()
        updates = dict()
        for trainer in trainers:
            model_hash = self._contract.previousUpdates(trainer)
            if model_hash != 0:
                updates[trainer] = model_hash
        return updates

    def _get_current_update_hashes(self, trainers=None):
        if trainers is None:
            trainers = self._contract.getTrainers()
        updates = dict()
        for trainer in trainers:
            model_hash = self._contract.currentUpdates(trainer)
            updates[trainer] = model_hash
        return updates

    def _get_models(self, model_hashes):
        models = dict()
        for trainer, model_hash in model_hashes.items():
            models[trainer] = self._ipfs_client.get_model(model_hash, self._model_constructor)
        return models

    def _get_genesis_model(self):
        gen_hash = self._contract.genesis()
        return self._ipfs_client.get_model(gen_hash, self._model_constructor)

    def _avg_model(self, models):
        avg_model = self._model_constructor()
        with torch.no_grad():
            for params in avg_model.parameters():
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model

    def _characteristic_function(self, *players):
        """
        The characteristic function used to calculate Shapley Value.
        The Shapley Value of a coalition of trainers is the marginal loss reduction
        of the average of their models
        """
        start_model = self._get_global_model()
        start_loss, _ = self._evaluate_model(start_model)
        hashes = self._get_current_update_hashes(players)
        models = self._get_models(hashes).values()
        avg_model = self._avg_model(models)
        loss, _ = self._evaluate_model(avg_model)
        return start_loss - loss
        