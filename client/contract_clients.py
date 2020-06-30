import json

import base58
from web3 import HTTPProvider, Web3


class _BaseContractClient:
    """
    Contains common features of both contract clients.

    Handles contract setup, conversions to and from bytes32 and other utils.
    """

    PROVIDER_ADDRESS = "http://127.0.0.1:7545"
    NETWORK_ID = "5777"
    IPFS_HASH_PREFIX = bytes.fromhex('1220')

    def __init__(self, contract_json_path, account_idx, address, deploy):
        self._w3 = Web3(HTTPProvider(self.PROVIDER_ADDRESS))

        self.address = self._w3.eth.accounts[account_idx]
        self._w3.eth.defaultAccount = self.address

        self._contract_json_path = contract_json_path

        self._contract, self.contract_address = self._instantiate_contract(address, deploy)


    def wait_for_tx(self, tx_hash):
        receipt = self._w3.eth.waitForTransactionReceipt(tx_hash)
        return receipt

    def _instantiate_contract(self, address=None, deploy=False):
        with open(self._contract_json_path) as json_file:
            crt_json = json.load(json_file)
            abi = crt_json['abi']
            bytecode = crt_json['bytecode']
            if address is None:
                if deploy:
                    tx_hash = self._w3.eth.contract(
                        abi=abi,
                        bytecode=bytecode
                    ).constructor().transact()
                    tx_receipt = self.wait_for_tx(tx_hash)
                    address = tx_receipt.contractAddress
                else:
                    address = crt_json['networks'][self.NETWORK_ID]['address']
        instance = self._w3.eth.contract(
            abi=abi,
            address=address
        )
        return instance, address

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


class CrowdsourceContractClient(_BaseContractClient):
    """
    Wrapper over the Crowdsource.sol ABI, to gracefully bridge Python data to Solidity.

    The API of this class should match that of the smart contract.
    """

    def __init__(self, account_idx, address, deploy):
        super().__init__(
            "build/contracts/Crowdsource.json",
            account_idx,
            address,
            deploy
        )

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

    def secondsRemaining(self):
        return self._contract.functions.secondsRemaining().call()

    def countTokens(self, address=None):
        if address is None:
            return self._contract.functions.countTokens().call()
        return self._contract.functions.countTokens(address).call()

    def countTotalTokens(self):
        return self._contract.functions.countTotalTokens().call()

    def madeContribution(self, address, training_round):
        return self._contract.functions.madecontribution(address, training_round).call()

    def setGenesis(self, model_cid, round_duration, max_num_updates):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setGenesis(
            cid_bytes, round_duration, max_num_updates).call()
        return self._contract.functions.setGenesis(cid_bytes, round_duration, max_num_updates).transact()

    def addModelUpdate(self, model_cid, training_round):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.addModelUpdate(
            cid_bytes, training_round).call()
        return self._contract.functions.addModelUpdate(
            cid_bytes, training_round).transact()

    def setTokens(self, model_cid, num_tokens):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setTokens(cid_bytes, num_tokens).call()
        return self._contract.functions.setTokens(
            cid_bytes, num_tokens).transact()


class ConsortiumContractClient(_BaseContractClient):
    """
    Wrapper over the Consortium.sol ABI, to gracefully bridge Python data to Solidity.

    The API of this class should match that of the smart contract.
    """

    def __init__(self, account_idx, address, deploy):
        super().__init__(
            "build/contracts/Consortium.json",
            account_idx,
            address,
            deploy
        )

    def main(self):
        return self._contract.functions.main().call()

    def auxiliaries(self):
        return self._contract.functions.auxiliaries().call()

    def countTokens(self, address=None):
        if address is None:
            return self._contract.functions.countTokens().call()
        return self._contract.functions.countTokens(address).call()

    def countTotalTokens(self):
        return self._contract.functions.countTotalTokens().call()

    def setGenesis(self, model_cid, round_duration, num_trainers):
        cid_bytes = self._to_bytes32(model_cid)
        self._contract.functions.setGenesis(
            cid_bytes, round_duration, num_trainers).call()
        return self._contract.functions.setGenesis(cid_bytes, round_duration, num_trainers).transact()

    def addAux(self, evaluator):
        self._contract.functions.addAux(evaluator).call()
        return self._contract.functions.addAux(evaluator).transact()
