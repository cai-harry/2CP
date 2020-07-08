# 2CP

My MSc Individual Project for Summer 2020.

This repository contains `2CP`, the software framework which implements the _Crowdsource Protocol_ and the _Consortium Protocol_.

## Project paper
[Blockchains for shared ownership and governance of Federated Learning models](https://1drv.ms/b/s!Ag6l4RY3j8vthYUVJIWq1XYcFK2D-g?e=B80G7n)

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

#### Adding `/client` to `PYTHONPATH`

Required to run unit tests or experiments. This is really easy with conda. From repo root:

```
conda develop client
```

VSCode should already be set up correctly; see `.vscode/settings.json` and `.env`. (Followed [these instructions](https://binx.io/blog/2020/03/05/setting-python-source-folders-vscode/))

### Setting up IPFS client
Requirements
- [IPFS Desktop](https://github.com/ipfs-shipyard/ipfs-desktop)

### Setting up solidity frameworks
Requirements
- [Truffle](https://www.trufflesuite.com/truffle)
- [Ganache](https://www.trufflesuite.com/ganache)

#### Setting up simulated blockchain
1. Open the Ganache app.
2. Quick start or set up a new workspace.
3. (Optional, to create a more realistic blockchain) In settings, under Server, turn off automine and set mining block time to 15 seconds (or set to 3 or 4 to make unit tests run more quickly). Need to do this to test behaviour when transactions aren't mined instantaneously.

### Running the unit tests
1. Spin up a blockchain as above.
2. Start up an IPFS node by opening the IPFS desktop app.
3. Compile the contracts: `truffle compile` (no need to deploy, the unit test does this automatically)
4. From repo root: `pytest -s` (the `-s` flag displays print statements.)
