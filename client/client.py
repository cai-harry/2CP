import functools
import methodtools
import os
import json

import base58
import ipfshttpclient
import matplotlib.pyplot as plt
import syft as sy
from syft.federated.floptimizer import Optims
import torch
from torch import nn, optim
from torch.nn import functional as F
from web3 import Web3, HTTPProvider

import shapley

_hook = sy.TorchHook(torch)


class IPFSClient:
    def __init__(self):
        self._ipfs_client = ipfshttpclient.connect(
            '/ip4/127.0.0.1/tcp/5001/http')

    def get_model(self, model_cid, model_constructor):
        model = model_constructor()
        with self._ipfs_client as ipfs:
            # saves to current directory, filename is model_cid
            ipfs.get(model_cid)
        model.load_state_dict(torch.load(model_cid))
        os.remove(model_cid)
        return model

    def add_model(self, model):
        model_filename = "tmp.pt"
        with self._ipfs_client as ipfs:
            torch.save(model.state_dict(), model_filename)
            res = ipfs.add(model_filename)
        os.remove(model_filename)
        model_cid = res['Hash']
        return model_cid


class ContractClient:
    """
    Wrapper over the Smart Contract ABI, to gracefully bridge Python data to Solidity.

    The API of this class should match that of the smart contract, perhaps with some extra utility functions.
    """

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    NETWORK_ID = "5777"
    CONTRACT_JSON_PATH = "build/contracts/Crowdsource.json"
    IPFS_HASH_PREFIX = bytes.fromhex('1220')

    def __init__(self, account_idx, address=None):
        self._web3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))
        self._contract = self._instantiate_contract(address)
        
        self.address = self._web3.eth.accounts[account_idx]
        self._web3.eth.defaultAccount = self.address

    def evaluator(self):
        return self._contract.functions.evaluator().call()

    def genesis(self):
        cid_bytes = self._contract.functions.genesis().call()
        return self._from_bytes32(cid_bytes)

    def updates(self, training_round):
        cid_bytes = self._contract.functions.updates(training_round).call()
        return [self._from_bytes32(b) for b in cid_bytes]

    def currentRound(self):
        return self._contract.functions.currentRound().call()

    def countTokens(self, address=None):
        if address is None:
            return self._contract.functions.countTokens().call()
        return self._contract.functions.countTokens(address).call()

    def countTotalTokens(self):
        return self._contract.functions.countTotalTokens().call()

    def setGenesis(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        return self._contract.functions.setGenesis(cid_bytes).transact()

    def addModelUpdate(self, model_cid, training_round):
        cid_bytes = self._to_bytes32(model_cid)
        return self._contract.functions.addModelUpdate(
            cid_bytes, training_round).transact()

    def setTokens(self, model_cid, num_tokens):
        cid_bytes = self._to_bytes32(model_cid)
        return self._contract.functions.setTokens(
            cid_bytes, num_tokens).transact()

    def wait_for_tx(self, tx_hash):
        return self._web3.eth.waitForTransactionReceipt(tx_hash)

    def _instantiate_contract(self, address=None):
        with open(self.CONTRACT_JSON_PATH) as json_file:
            crt_json = json.load(json_file)
            abi = crt_json['abi']
            if address is None:
                address = crt_json['networks'][self.NETWORK_ID]['address']
        instance = self._web3.eth.contract(
            abi=abi,
            address=address
        )
        return instance

    def _to_bytes32(self, model_cid):
        bytes34 = base58.b58decode(model_cid)
        assert bytes34[:2] == self.IPFS_HASH_PREFIX, \
            f"IPFS cid should begin with {self.IPFS_HASH_PREFIX} but got {bytes34[:2].hex()}"
        bytes32 = bytes34[2:]
        return bytes32

    def _from_bytes32(self, bytes32):
        bytes34 = self.IPFS_HASH_PREFIX + bytes32
        model_cid = base58.b58encode(bytes34).decode()
        return model_cid

class CrowdsourceClient:
    # TODO: BaseClient -> TrainerClient, EvaluatorClient -> CrowdsourceClient

    TOKENS_PER_UNIT_LOSS = 1e18  # number of wei per ether

    def __init__(self, name, data, targets, model_constructor, account_idx, contract_address=None):
        self.name = name
        self.data_length = min(len(data), len(targets))

        self._worker = sy.VirtualWorker(_hook, id=name)
        self._data = data.send(self._worker)
        self._targets = targets.send(self._worker)
        self._data_loader = torch.utils.data.DataLoader(
            sy.BaseDataset(self._data, self._targets),
            batch_size=len(data),
            shuffle=True
        )
        self._model_constructor = model_constructor
        self._contract = ContractClient(account_idx, contract_address)
        self._ipfs_client = IPFSClient()

    def is_evaluator(self):
        return self._contract.evaluator() == self._contract.address

    def get_token_count(self):
        return self._contract.countTokens(), self._contract.countTotalTokens()

    def get_current_global_model(self):
        """
        Calculate, or get from cache, the current global model.
        """
        current_training_round = self._contract.currentRound()
        return self._get_global_model(current_training_round), current_training_round

    def set_genesis_model(self):
        """
        Create, upload and record the genesis model.
        """
        genesis_model = self._model_constructor()
        genesis_cid = self._ipfs_client.add_model(genesis_model)
        tx = self._contract.setGenesis(genesis_cid)
        return tx

    def run_training_round(self, epochs, learning_rate):
        """
        Run training using own data, upload and record the contribution.
        """
        model, training_round = self.get_current_global_model()
        model = self._train_model(model, epochs, learning_rate)
        uploaded_cid = self._upload_model(model)
        tx = self._record_model(uploaded_cid, training_round + 1)
        return tx

    def evaluate_updates(self, training_round):
        """
        Provide Shapley Value score for each update in the given training round.
        """
        cids = self._get_cids(training_round)
        def characteristic_function(*c):
            return self._marginal_value(training_round, *c)
        scores = shapley.values(
            characteristic_function, cids)
        return scores

    def set_tokens(self, cid_scores):
        """
        Record the given Shapley value scores for the given contributions.
        """
        txs = []
        for cid, score in cid_scores.items():
            num_tokens = max(0, int(score * self.TOKENS_PER_UNIT_LOSS))
            tx = self._contract.setTokens(cid, num_tokens)
            txs.append(tx)
        return txs

    def evaluate_current_global(self):
        """
        Evaluate the current global model using own data.
        """
        current_training_round = self._contract.currentRound()
        return self._evaluate_global(current_training_round)

    def predict(self):
        model, _ = self.get_current_global_model()
        model = model.send(self._worker)
        predictions = []
        with torch.no_grad():
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = model(data).get()
                predictions.append(pred)
        return torch.stack(predictions)

    def wait_for(self, txs):
        receipts = []
        if txs:
            for tx in txs:
                receipts.append(self._contract.wait_for_tx(tx))
            txs.clear()  # clears the list of pending transactions passed in as argument
        return receipts

    @methodtools.lru_cache()
    def _get_global_model(self, training_round):
        """
        Calculate global model at the the given training round by aggregating updates from previous round.

        This can only be done if training_round matches the current contract training round.
        """
        model_cids = self._get_cids(training_round-1)
        models = self._get_models(model_cids)
        avg_model = self._avg_model(models)
        return avg_model

    @methodtools.lru_cache()
    def _evaluate_global(self, training_round):
        """
        Evaluate the global model at the given training round.
        """
        model = self._get_global_model(training_round)
        loss, accuracy = self._evaluate_model(model)
        return loss, accuracy

    def _train_model(self, model, epochs, lr):
        model = model.send(self._worker)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                optimizer.zero_grad()
                pred = model(data)
                loss = F.mse_loss(pred, labels)
                loss.backward()
                optimizer.step()
        return model.get()

    def _evaluate_model(self, model):
        model = model.send(self._worker)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = model(data)
                total_loss += F.mse_loss(pred, labels)
                total_correct += (torch.round(pred) == labels).float().sum()
        avg_loss = total_loss / len(self._data_loader)
        accuracy = total_correct / self.data_length
        model = model.get()
        return avg_loss.get().item(), accuracy.get().item()

    def _upload_model(self, model):
        """Uploads the given model to IPFS."""
        uploaded_cid = self._ipfs_client.add_model(model)
        return uploaded_cid

    def _record_model(self, uploaded_cid, training_round):
        """Records the given model IPFS cid on the smart contract."""
        return self._contract.addModelUpdate(uploaded_cid, training_round)

    def _get_cids(self, training_round):
        if training_round < 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 0:
            return [self._contract.genesis()]
        cids = self._contract.updates(training_round)
        if not cids:  # if cids is empty, refer to previous round
            return self._get_cids(training_round - 1)
        return cids

    def _get_models(self, model_cids):
        models = []
        for cid in model_cids:
            model = self._ipfs_client.get_model(cid, self._model_constructor)
            models.append(model)
        return models

    def _get_genesis_model(self):
        gen_cid = self._contract.genesis()
        return self._ipfs_client.get_model(gen_cid, self._model_constructor)

    def _avg_model(self, models):
        avg_model = self._model_constructor()
        with torch.no_grad():
            for params in avg_model.parameters():
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model

    @functools.lru_cache()
    def _marginal_value(self, training_round, *update_cids):
        """
        The characteristic function used to calculate Shapley Value.
        The Shapley Value of a coalition of trainers is the marginal loss reduction
        of the average of their models
        """
        start_loss, _ = self._evaluate_global(training_round)
        models = self._get_models(update_cids)
        avg_model = self._avg_model(models)
        loss, _ = self._evaluate_model(avg_model)
        return start_loss - loss
