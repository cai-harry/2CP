import functools
import methodtools
import json

import base58
from web3 import Web3, HTTPProvider

from client import IPFSClient, ContractClient, CrowdsourceClient


class ConsortiumContractClient():
    # TODO: both contract clients should inherit from a base class

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    NETWORK_ID = "5777"
    CONTRACT_JSON_PATH = "build/contracts/Consortium.json"
    IPFS_HASH_PREFIX = bytes.fromhex('1220')

    def __init__(self, account_idx, address=None):
        self._web3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))
        self._contract = self._instantiate_contract(address)

        self.address = self._web3.eth.accounts[account_idx]
        self._web3.eth.defaultAccount = self.address

    def main(self):
        return self._contract.functions.main().call()

    def subs(self):
        return self._contract.functions.subs().call()

    def countTokens(self, address=None):
        if address is None:
            return self._contract.functions.countTokens().call()
        return self._contract.functions.countTokens(address).call()

    def countTotalTokens(self):
        return self._contract.functions.countTotalTokens().call()

    def setGenesis(self, model_cid):
        cid_bytes = self._to_bytes32(model_cid)
        return self._contract.functions.setGenesis(cid_bytes).transact()

    def addSub(self, evaluator):
        return self._contract.functions.addSub(evaluator).transact()

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


class ConsortiumCreatorClient():
    # TODO: BaseClient -> ConsortiumCreatorClient()
    def __init__(self, name, model_constructor, account_idx, contract_address=None):
        self.name = name

        self._model_constructor = model_constructor
        self._contract = ConsortiumContractClient(
            account_idx, contract_address)
        self._ipfs_client = IPFSClient()

    def set_genesis_model(self):
        """
        Create, upload and record the genesis model.
        """
        genesis_model = self._model_constructor()
        genesis_cid = self._ipfs_client.add_model(genesis_model)
        tx = self._contract.setGenesis(genesis_cid)
        return tx

    def add_sub(self, evaluator):
        return self._contract.addSub(evaluator)

    def wait_for(self, txs):
        receipts = []
        if txs:
            for tx in txs:
                receipts.append(self._contract.wait_for_tx(tx))
            txs = []  # clears the list of pending transactions passed in as argument
        return receipts


class ConsortiumClient():
    # TODO: BaseClient -> ConsortiumClient
    def __init__(self, name, data, targets, model_constructor, account_idx):
        self.name = name

        self._data = data
        self._targets = targets
        self._model_constructor = model_constructor
        self._account_idx = account_idx

        self._contract = ConsortiumContractClient(account_idx)
        self.address = self._contract.address

        self._main_client = CrowdsourceClient(name + " (main)", data, targets, model_constructor, account_idx,
                                              contract_address=self._contract.main())

    def get_token_count(self):
        return self._contract.countTokens(), self._contract.countTotalTokens()

    def get_current_global_model(self):
        return self._main_client.get_current_global_model()

    def run_training_round(self, epochs, learning_rate):
        train_clients = self._get_train_clients()
        return [
            client.run_training_round(epochs, learning_rate)
            for client in train_clients
        ]

    def evaluate_updates(self, training_round):
        eval_clients = self._get_eval_clients()
        return [
            client.evaluate_updates(training_round)
            for client in eval_clients
        ]

    def set_tokens(self, cid_scores):
        eval_clients = self._get_eval_clients()
        txs = []
        for scores, client in zip(cid_scores, eval_clients):
            txs.extend(client.set_tokens(scores))
        return txs

    def evaluate_current_global(self):
        return self._main_client.evaluate_current_global()

    def predict(self):
        return self._main_client.predict()

    def wait_for(self, txs):
        # TODO: should inherit this
        self._main_client.wait_for(txs)   # doesn't matter which client we use

    def _get_sub_clients(self):
        return [
            CrowdsourceClient(self.name + f" (sub {i})", self._data, self._targets, self._model_constructor, self._account_idx,
                              contract_address=sub)
            for i, sub in enumerate(self._contract.subs())]

    def _get_train_clients(self):
        sub_clients = self._get_sub_clients()
        train_clients = [
            sub for sub in sub_clients if not sub.is_evaluator()
        ]
        train_clients.append(self._main_client)
        return train_clients

    def _get_eval_clients(self):
        sub_clients = self._get_sub_clients()
        return [
            sub for sub in sub_clients if sub.is_evaluator()]

