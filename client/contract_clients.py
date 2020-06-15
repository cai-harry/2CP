import json

import base58
from web3 import HTTPProvider, Web3


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

