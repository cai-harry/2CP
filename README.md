# BlockchainFL

My MSc Individual Project for Summer 2020.

## Project paper
[Blockchains for shared ownership and governance of Federated Learning models](https://www.overleaf.com/read/bjznxpcbxvfs)

## Development

### Setting up python environment
Requirements
- Conda
```
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

### Setting up solidity frameworks
Requirements
- Truffle
- Ganache

### Running `test_client.py`
1. Compile and deploy contracts: `truffle migrate --reset`
2. Copy deployed contract address.
3. From repo root: `python clients/test_client.py`
4. Paste the deployed contract address when prompted.
