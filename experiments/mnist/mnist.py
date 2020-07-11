import argparse
import json
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from clients import CrowdsourceClient, ConsortiumSetupClient, ConsortiumClient
from utils import print_global_performance, print_token_count


class Data:
    def __init__(self, train, subset=False, exclude_digits=None):
        self._dataset = datasets.MNIST(
            'experiments/mnist/resources',
            train=train)
        self.data = self._dataset.data.float().view(-1, 1, 28, 28) / 255
        self.targets = self._dataset.targets
        if subset:
            d, t, *_ = self.split(200)
            self.data = torch.cat(d)
            self.targets = torch.cat(t)
        if exclude_digits is not None:
            for digit in exclude_digits:
                mask = self.targets != digit
                self.data = self.data[mask]
                self.targets = self.targets[mask]

    def split(self, n, ratios=None, flip_probs=None):
        """
        Splits the dataset into n chunks with the given ratios and label flip probabilities.
        """
        if ratios is None:
            ratios = [1] * n
        if flip_probs is None:
            flip_probs = [0] * n
        if not n == len(ratios) == len(flip_probs):
            raise ValueError(f"Lengths of input arguments must match n={n}")
        perm = torch.randperm(len(self.data))
        num_chunks = sum(ratios)
        chunks = torch.chunk(perm, num_chunks)

        chunk_it = 0
        data = []
        targets = []
        for r, p in zip(ratios, flip_probs):
            include_chunks = list(range(chunk_it, chunk_it+r))
            idxs = torch.cat([chunks[idxs] for idxs in include_chunks])
            chunk_it += r

            d = torch.index_select(input=self.data, dim=0, index=idxs)
            t = self._flip_targets(
                torch.index_select(
                    input=self.targets,
                    dim=0,
                    index=idxs
                ), p)

            data.append(d)
            targets.append(t)
        return data, targets

    def _flip_targets(self, targets, flip_p):
        flip_num = int(flip_p * len(targets))
        flip_idx = torch.randperm(len(targets))[:flip_num]
        targets[flip_idx] = torch.randint(
            low=0, high=10, size=(len(flip_idx),))
        return targets


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _make_clients(split_type, num_trainers, ratios, flip_probs, unique_digits, protocol):
    if protocol == 'crowdsource':
        alice_dataset = Data(train=False, subset=QUICK_RUN)
        alice_data = alice_dataset.data
        alice_targets = alice_dataset.targets
        alice = CrowdsourceClient(
            name="Alice",
            data=alice_data,
            targets=alice_targets,
            model_constructor=Model,
            model_criterion=F.nll_loss,
            account_idx=0,
            contract_address=None,
            deploy=True
        )
        trainers = _make_trainers(
            split_type=split_type,
            num_trainers=num_trainers,
            ratios=ratios,
            flip_probs=flip_probs,
            unique_digits=unique_digits,
            client_constructor=CrowdsourceClient,
            contract_address=alice.contract_address)
        alice.set_genesis_model(
            round_duration=ROUND_DURATION,
            max_num_updates=len(trainers)
        )
    elif protocol == 'consortium':
        alice = ConsortiumSetupClient(
            name="Alice",
            model_constructor=Model,
            account_idx=0,
            contract_address=None,
            deploy=True
        )
        trainers = _make_trainers(
            split_type=split_type,
            num_trainers=num_trainers,
            ratios=ratios,
            flip_probs=flip_probs,
            unique_digits=unique_digits,
            client_constructor=ConsortiumClient,
            contract_address=alice.contract_address)
        alice.set_genesis_model(
            round_duration=ROUND_DURATION,
            max_num_updates=len(trainers)
        )
        alice.add_auxiliaries([
            trainer.address for trainer in trainers
        ])
    return alice, trainers


def _make_trainers(
        split_type,
        num_trainers,
        ratios,
        flip_probs,
        unique_digits,
        client_constructor,
        contract_address):
    # instantiate data
    if split_type == 'equal':
        data, targets = Data(train=True, subset=QUICK_RUN).split(num_trainers)
    if split_type == 'size':
        data, targets = Data(train=True, subset=QUICK_RUN).split(
            num_trainers, ratios=ratios)
    if split_type == 'flip':
        data, targets = Data(train=True, subset=QUICK_RUN).split(
            num_trainers, flip_probs=flip_probs)
    if split_type == 'unique_digits':
        dataset_unique = Data(train=True, subset=QUICK_RUN,
                              exclude_digits=set(range(10))-set(unique_digits))
        data_unique = dataset_unique.data
        targets_unique = dataset_unique.targets
        data_others, targets_others = Data(
            train=True, subset=QUICK_RUN, exclude_digits=unique_digits
        ).split(num_trainers-1)
        data = [data_unique] + data_others
        targets = [targets_unique] + targets_others

    # instantiate clients
    common_args = {
        'model_constructor': Model,
        'model_criterion': F.nll_loss,
        'contract_address': contract_address,
        'deploy': False
    }

    trainers = []
    names = ["Bob", "Carol", "David", "Eve",
             "Frank", "Georgia", "Henry", "Isabel", "Joe"]
    for i, name, d, t in zip(range(num_trainers), names, data, targets):
        trainer = client_constructor(
            name=name,
            data=d,
            targets=t,
            account_idx=i+1,
            **common_args
        )
        trainers.append(trainer)
    return trainers


def _global_accuracy(client):
    """
    A hacky function that gets the accuracy rather than the loss.
    """
    output = client.predict()
    pred = output.argmax(dim=2, keepdim=True).squeeze()
    num_correct = (pred == client._targets).float().sum().item()
    accuracy = num_correct / len(pred)
    return accuracy


def _token_count_histories(trainers):
    token_count_history_by_name = {}
    for trainer in trainers:
        token_count_history_by_name[trainer.name] = [
            trainer.get_token_count(training_round=i)
            for i in range(1, TRAINING_ITERATIONS+1)
        ]
    return token_count_history_by_name


def _save_results(results):
    filedir = "experiments/mnist/results/"
    if QUICK_RUN:
        filedir += "quick/"
    filename = "all.json"
    filepath = filedir + filename
    with open(filepath) as f:
        all_results = json.load(f)
    all_results.append(results)
    with open(filepath, 'w') as f:
        json.dump(all_results, f,
                  indent=4)
    print(f"{filepath} now has {len(all_results)} results")


def run_experiment(
    split_type,
    protocol,
    eval_method,
    seed,
    num_trainers=3,
    ratios=None,
    flip_probs=None,
    unique_digits=None
):

    # check args
    if split_type not in {'equal', 'size', 'flip', 'unique_digits'}:
        raise KeyError(f"split_type={split_type} is not a valid option")
    if protocol not in {'crowdsource', 'consortium'}:
        raise KeyError(f"protocol={protocol} is not a valid option")
    if not 2 <= num_trainers <= 9:
        raise ValueError(
            f"Expected num_trainers to be between 2 and 9, got {num_trainers}")

    # make results dict, add details of current experiment
    results = {}
    results['split_type'] = split_type
    results['protocol'] = protocol
    results['eval_method'] = eval_method
    results['seed'] = seed
    results['num_trainers'] = num_trainers
    if ratios is not None:
        results['ratios'] = ratios
    if flip_probs is not None:
        results['flip_probs'] = flip_probs
    if unique_digits is not None:
        results['unique_digits'] = unique_digits

    # set up
    torch.manual_seed(seed)
    alice, trainers = _make_clients(
        split_type, num_trainers, ratios, flip_probs, unique_digits, protocol)
    results['contract_address'] = alice.contract_address
    results['trainers'] = [trainer.name for trainer in trainers]

    # define training threads
    threads = [
        threading.Thread(
            target=trainer.train_until,
            kwargs=TRAINING_HYPERPARAMS,
            daemon=True
        ) for trainer in trainers
    ]

    # define evaluation threads
    if protocol == 'crowdsource':
        threads.append(
            threading.Thread(
                target=alice.evaluate_until,
                args=(TRAINING_ITERATIONS, eval_method),
                daemon=True
            )
        )
    if protocol == 'consortium':
        threads.extend([
            threading.Thread(
                target=trainer.evaluate_until,
                args=(TRAINING_ITERATIONS, eval_method),
                daemon=True
            ) for trainer in trainers
        ])

    # run all threads in parallel
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # get results and record them
    if protocol == 'crowdsource':
        results['global_loss'] = [alice.evaluate_global(i)
                                  for i in range(1, TRAINING_ITERATIONS+2)]
        results['final_global_accuracy'] = _global_accuracy(alice)
    results['token_counts'] = _token_count_histories(trainers)
    results['total_token_counts'] = [alice.get_total_token_count(i)
                                     for i in range(1, TRAINING_ITERATIONS+1)]

    _save_results(results)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiments on the MNIST dataset using 2CP.")
    parser.add_argument(
        '--full',
        help='Do a full run. Otherwise, does a quick run for testing purposes during development.',
        action='store_true'
    )
    args = parser.parse_args()

    QUICK_RUN = not args.full
    if QUICK_RUN:
        TRAINING_ITERATIONS = 2
    else:
        TRAINING_ITERATIONS = 3
    TRAINING_HYPERPARAMS = {
        'final_round_num': TRAINING_ITERATIONS,
        'batch_size': 32,
        'epochs': 1,
        'learning_rate': 1e-2
    }
    ROUND_DURATION = 1000  # should always end early

    if QUICK_RUN:
        experiments = [
            {'split_type': 'equal', 'num_trainers': 3},
            {'split_type': 'size',  'ratios': [1, 4, 4]},
            {'split_type': 'flip',  'flip_probs': [0.3, 0, 0]},
            {'split_type': 'unique_digits',
                'unique_digits': [9], 'num_trainers': 3},
        ]
        method = 'step'
        seed = 88
        for exp in experiments:
            for protocol in ['crowdsource', 'consortium']:
                run_experiment(protocol=protocol, eval_method=method, seed=seed, **exp)
    else:
        experiments = [
            {'split_type': 'equal', 'num_trainers': 2},
            {'split_type': 'equal', 'num_trainers': 3},
            {'split_type': 'equal', 'num_trainers': 4},
            {'split_type': 'equal', 'num_trainers': 5},
            {'split_type': 'size',  'ratios': [1, 4, 4]},
            {'split_type': 'size',  'ratios': [1, 2, 2]},
            {'split_type': 'size',  'ratios': [2, 1, 1]},
            {'split_type': 'size',  'ratios': [4, 1, 1]},
            {'split_type': 'flip',  'flip_probs': [0.1, 0, 0]},
            {'split_type': 'flip',  'flip_probs': [0.2, 0, 0]},
            {'split_type': 'flip',  'flip_probs': [0.3, 0, 0]},
            {'split_type': 'flip',  'flip_probs': [0.4, 0, 0]},
            {'split_type': 'flip',  'flip_probs': [0.5, 0, 0]},
            {'split_type': 'unique_digits',
                'unique_digits': [9], 'num_trainers': 2},
            {'split_type': 'unique_digits', 'unique_digits': [
                7, 8, 9], 'num_trainers': 2},
            {'split_type': 'unique_digits', 'unique_digits': [
                5, 6, 7, 8, 9], 'num_trainers': 2},
            {'split_type': 'unique_digits',
                'unique_digits': [9], 'num_trainers': 3},
            {'split_type': 'unique_digits', 'unique_digits': [
                5, 6, 7, 8, 9], 'num_trainers': 3}
        ]
        method = 'step'
        for seed in [32, 76, 88]:
            for exp in experiments:
                for protocol in ['crowdsource', 'consortium']:
                    run_experiment(protocol=protocol, eval_method=method, seed=seed, **exp)
