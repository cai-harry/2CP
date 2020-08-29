# 2CP

My MSc Individual Project for Summer 2020.

## Project paper
[2CP: Decentralized protocols to transparently evaluate contributivity in Blockchain Federated Learning environments](https://www.overleaf.com/project/5e7f295512360300014df284)

## What's in this repo?

`2cp/`: This repository contains `2CP`, the software framework which implements the _Crowdsource Protocol_ and the _Consortium Protocol_.

`build/`: The compiled contracts. Remember to update these when you make changes to the smart contracts.

`contracts/`: The solidity source code for `2CP` smart contracts.

`experiments/`: It also contains the code used to run the experiments described in the project paper, and the corresponding results.

`migrations/`: Deployment scripts used by Truffle.

`scripts/`: Miscellaneous scripts to tidy up results files, produce plots, etc.

`tests/`: Unit tests for `2CP`.

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

#### Adding `/2cp` to `PYTHONPATH`

Required to run unit tests or experiments. This is really easy with conda. From repo root:

```
conda develop 2cp
```

VSCode should already be set up correctly; see `.vscode/settings.json` and `.env`. (Followed [these instructions](https://binx.io/blog/2020/03/05/setting-python-source-folders-vscode/))

### Setting up IPFS client
Requirements
- [IPFS Desktop](https://github.com/ipfs-shipyard/ipfs-desktop)

### Setting up solidity frameworks
Requirements
- [Truffle](https://www.trufflesuite.com/truffle)
- [Ganache](https://www.trufflesuite.com/ganache)

### Setting up simulated blockchain
1. Open the Ganache app.
2. Quick start or set up a new workspace.
3. (Optional, to create a more realistic blockchain) In settings, under Server, turn off automine and set mining block time to 15 seconds (or set to 3 or 4 to make unit tests run more quickly). Need to do this to test behaviour when transactions aren't mined instantaneously.

### Alternative: Docker

Best option if not using Windows. Requires only [Docker](https://www.docker.com/get-started)

1. Build the docker image: (in repo root) `docker build -t 2cp .`
2. Launch the image in interactive mode: `docker run -it 2cp` 
3. Start the ipfs daemon in the background: `ipfs daemon &`
4. Start the blockchain in the background: `ganache-cli --port=7545 &`

### Running unit tests or experiments
1. Compile the contracts: `truffle compile` (2CP deploys them automatically when it runs)
2. Spin up a blockchain as above.
3. Start up an IPFS node by opening the IPFS desktop app.
4. Run one of these commands:
  - Unit tests: `pytest -s` (the `-s` flag displays print statements.)
  - MNIST experiments: `python experiments/mnist/mnist.py --full` (without the `--full` flag, will just run a quick version for debugging purposes.) 
  - COVID experiments: `python experiments/covid/covid.py --full`

Choose which experiments to run by defining changing the `experiments` dictionary in the `mnist.py` or `covid.py` scripts.
