# BlockchainFL

My MSc Individual Project for Summer 2020.

## Project paper
[Blockchains for shared ownership and governance of Federated Learning models](https://www.overleaf.com/read/bjznxpcbxvfs)

## Development

### Setting up python environment

Recommended:
- Anaconda (install for user)
- Visual Studio Code

It is easiest to create and use a fresh conda environment.
```
conda create -n proj
activate proj
```
VSCode should already be set up to use this `proj` conda environment.

Next, use conda to install pytorch as pip is not supported.
We need exactly version 1.4.0 for compatibility with PySyft.
```
conda install pytorch==1.4.0 torchvision -c pytorch
```

Use pip to install everything else.
```
pip install -r requirements.txt
```

### Setting up IPFS client
Requirements
- [IPFS Desktop](https://github.com/ipfs-shipyard/ipfs-desktop)

### Setting up solidity frameworks
Requirements
- [Truffle](https://www.trufflesuite.com/truffle)
- [Ganache](https://www.trufflesuite.com/ganache)

### Running `test_client.py`
1. Spin up a blockchain by opening the Ganache app. Default settings should work.
2. Start up an IPFS node by opening the IPFS desktop app.
3. Compile and deploy contracts: `truffle migrate --reset`
4. From repo root: `python clients/test_client.py`
