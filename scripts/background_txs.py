"""
Usage: `python background_txs.py` (^C to stop)

Makes regular, unrelated transactions to simulate a live blockchain with other transactions happening outside of the protocol.
Convenient for quality of life when using Ganache, as it triggers new blocks regularly even on Automine.
"""

import time

import contract_clients

FROM_IDX = 9
TO_IDX = 0
AMOUNT = 1
INTERVAL = 5


class EthClient(contract_clients.BaseEthClient):
    def sendTransaction(self, account_idx, amount):
        """
        Make a transaction of wei to the specified account.
        """
        recipient = self._w3.eth.accounts[account_idx]
        self._w3.eth.sendTransaction({
            'to': recipient,
            'value': amount
        })

    def blockNumber(self):
        """
        Get the current block number.
        """
        return self._w3.eth.blockNumber


if __name__ == "__main__":
    client = EthClient(FROM_IDX)
    try:
        while True:
            client.sendTransaction(TO_IDX, AMOUNT)
            print(f"Block number {client.blockNumber()}")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Stopping...")

