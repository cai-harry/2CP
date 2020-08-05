import joblib
import json
import threading
import time
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from pyvacy.analysis import moments_accountant as epsilon

from clients import CrowdsourceClient, ConsortiumSetupClient, ConsortiumClient




class _Data:
    def __init__(self):
        self.data = []
        self.targets = []

    def split(self, n, ratios=None, flip_probs=None, disjointness=0):
        """
        Splits the dataset into n chunks with the given ratios and label flip probabilities.

        n: Number of clients
        ratios: List of size ratios for the split
        flip_probs: List of proportions of label flipping
        disjointness: 0-1 disjointness proportion; 0 = random split, 1 = disjoint split by class
        """
        if ratios is None:
            ratios = [1] * n
        if flip_probs is None:
            flip_probs = [0] * n
        if not n == len(ratios) == len(flip_probs):
            raise ValueError(f"Lengths of input arguments must match n={n}")

        # sort indices by class
        sorted_targets, sorted_idxs = torch.sort(self.targets)
        perm = sorted_idxs.clone()

        print(f"\tsorted_targets={sorted_targets}")
        print(f"\t(counts)={torch.unique(sorted_targets, return_counts=True)}")

        # take sorted indices and shuffle a proportion of them
        shuffle_proportion = 1 - disjointness
        shuffle_num = int(shuffle_proportion * len(sorted_idxs))
        shuffle_idxs = torch.randperm(len(sorted_idxs))[:shuffle_num]
        sorted_shuffle_idxs, _ = torch.sort(shuffle_idxs)
        for i, j in zip(sorted_shuffle_idxs, shuffle_idxs):
            perm[i] = sorted_idxs[j]

        print(f"\tperm_targets={self.targets[perm]}")
        print(
            f"\t(counts)={torch.unique(self.targets[perm], return_counts=True)}")

        # split into chunks
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
        num_classes = len(torch.unique(targets))
        flip_num = int(flip_p * len(targets))
        flip_idx = torch.randperm(len(targets))[:flip_num]
        flipped_labels = torch.randint(
            low=0, high=num_classes, size=(len(flip_idx),), dtype=targets.dtype)
        for i, label in zip(flip_idx, flipped_labels):
            targets[i] = label
        return targets


class MNISTData(_Data):
    def __init__(self, train, subset=False, exclude_digits=None):
        self._dataset = datasets.MNIST(
            'experiments/mnist/resources',
            train=train)
        self.data = self._dataset.data.float().view(-1, 1, 28, 28) / 255
        self.targets = self._dataset.targets
        if subset:
            perm = torch.randperm(len(self.data))
            sub = perm[:(len(self.data)//200)]
            self.data = self.data[perm]
            self.targets = self.targets[perm]
        if exclude_digits is not None:
            for digit in exclude_digits:
                mask = self.targets != digit
                self.data = self.data[mask]
                self.targets = self.targets[mask]


class CovidData(_Data):
    def __init__(self, train, subset=False):
        # TODO: dataset is too small to subset, so subset argument is unused
        data_dir = "experiments/covid/resources/"
        if train:
            self._df = pd.read_csv(data_dir + "train.csv")
        else:
            self._df = pd.read_csv(data_dir + "test.csv")
        X, y = self._split_into_lists(
            data=self._df.drop(['ICU'], 1),
            labels=self._df['ICU']
        )
        self.data = self._convert_to_tensor(X)
        self.targets = self._convert_to_tensor(y)
        assert len(self.data) == len(self.targets), \
            f"Data and targets have different lengths {len(self.data)} and {len(self.targets)}"

    def _split_into_lists(self, data, labels):
        record_list = list()
        result_list = list()
        for _, row in data.iterrows():
            record_list.append(row)
        for row in labels:
            converted_label = float(row)
            result_list.append(converted_label)
        return record_list, result_list

    def _convert_to_tensor(self, data):
        tensors = []
        for record in data:
            tensors.append(torch.tensor(record))
        return torch.stack(tensors)


class MNISTModel(nn.Module):
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


class CovidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(62, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()


class ExperimentRunner:

    def __init__(
        self,
        quick_run,
        training_iterations,
        training_hyperparams,
        round_duration
    ):
        self.QUICK_RUN = quick_run
        self.TRAINING_ITERATIONS = training_iterations
        self.TRAINING_HYPERPARAMS = training_hyperparams
        self.ROUND_DURATION = round_duration

    def run_experiment(
        self,
        dataset,
        split_type,
        protocol,
        eval_method,
        seed,
        num_trainers=3,
        ratios=None,
        flip_probs=None,
        disjointness=0,
        dp_params=None,
        unique_digits=None
    ):

        # check args
        if split_type not in {'equal', 'size', 'flip', 'noniid', 'unique_digits', 'dp'}:
            raise KeyError(f"split_type={split_type} is not a valid option")
        if protocol not in {'crowdsource', 'consortium'}:
            raise KeyError(f"protocol={protocol} is not a valid option")
        if not 2 <= num_trainers <= 9:
            raise ValueError(
                f"Expected num_trainers to be between 2 and 9, got {num_trainers}")

        # make results dict, add details of current experiment
        results = {}
        results['dataset'] = dataset
        results['quick_run'] = self.QUICK_RUN
        results.update(self.TRAINING_HYPERPARAMS)
        results['split_type'] = split_type
        results['protocol'] = protocol
        results['eval_method'] = eval_method
        results['seed'] = seed
        results['num_trainers'] = num_trainers
        if ratios is not None:
            results['ratios'] = ratios
        if flip_probs is not None:
            results['flip_probs'] = flip_probs
        results['disjointness'] = disjointness
        if dp_params is not None:
            results.update(dp_params)
        if unique_digits is not None:
            results['unique_digits'] = unique_digits

        # set up
        torch.manual_seed(seed)
        alice, trainers = self._make_clients(
            dataset, split_type, num_trainers, ratios, flip_probs, disjointness, unique_digits, protocol)
        results['contract_address'] = alice.contract_address
        results['trainers'] = [trainer.name for trainer in trainers]
        if dataset == 'mnist':
            results['digit_counts'] = self._label_counts(10, trainers)
        if dataset == 'covid':
            results['label_counts'] = self._label_counts(2, trainers)
        if dp_params is not None:
            results['epsilon'] = [
                epsilon(
                    N=trainer.data_length,
                    batch_size=self.TRAINING_HYPERPARAMS['batch_size'],
                    noise_multiplier=dp_params['noise_multiplier'],
                    epochs=self.TRAINING_HYPERPARAMS['epochs'],
                    delta=dp_params['delta']
                )
                for trainer in trainers
            ]

        # define training threads
        if dp_params is not None:
            train_hyperparams = {
                **self.TRAINING_HYPERPARAMS,
                'dp_params': dp_params
            }
        else:
            train_hyperparams = self.TRAINING_HYPERPARAMS

        threads = [
            threading.Thread(
                target=trainer.train_until,
                kwargs=train_hyperparams,
                daemon=True
            ) for trainer in trainers
        ]

        # define evaluation threads
        if protocol == 'crowdsource':
            threads.append(
                threading.Thread(
                    target=alice.evaluate_until,
                    args=(self.TRAINING_ITERATIONS, eval_method),
                    daemon=True
                )
            )
        if protocol == 'consortium':
            threads.extend([
                threading.Thread(
                    target=trainer.evaluate_until,
                    args=(self.TRAINING_ITERATIONS, eval_method),
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
                                      for i in range(1, self.TRAINING_ITERATIONS+2)]
            one_hot_output = (dataset == 'mnist')
            results['final_global_accuracy'] = self._global_accuracy(
                alice, one_hot_output)
        results['token_counts'] = self._token_count_histories(trainers)
        results['total_token_counts'] = [alice.get_total_token_count(i)
                                         for i in range(1, self.TRAINING_ITERATIONS+1)]
        results['gas_used'] = self._gas_used(alice, trainers)

        self._save_results(results)

        return results

    def _make_clients(
        self,
        dataset,
        split_type,
        num_trainers,
        ratios,
        flip_probs,
        disjointness,
        unique_digits,
        protocol
    ):
        if dataset == 'mnist':
            dataset_constructor = MNISTData
            model_constructor = MNISTModel
            model_criterion = F.nll_loss
        elif dataset == 'covid':
            dataset_constructor = CovidData
            model_constructor = CovidModel
            model_criterion = F.mse_loss
        else:
            raise KeyError(f"Invalid dataset key: {dataset}")
        alice_dataset = dataset_constructor(train=False, subset=self.QUICK_RUN)
        alice_data = alice_dataset.data
        alice_targets = alice_dataset.targets

        if protocol == 'crowdsource':
            alice = CrowdsourceClient(
                name="Alice",
                data=alice_data,
                targets=alice_targets,
                model_constructor=model_constructor,
                model_criterion=model_criterion,
                account_idx=0,
                contract_address=None,
                deploy=True
            )
            trainers = self._make_trainers(
                dataset_constructor=dataset_constructor,
                model_constructor=model_constructor,
                model_criterion=model_criterion,
                split_type=split_type,
                num_trainers=num_trainers,
                ratios=ratios,
                flip_probs=flip_probs,
                disjointness=disjointness,
                unique_digits=unique_digits,
                client_constructor=CrowdsourceClient,
                contract_address=alice.contract_address)
            alice.set_genesis_model(
                round_duration=self.ROUND_DURATION,
                max_num_updates=len(trainers)
            )
        elif protocol == 'consortium':
            alice = ConsortiumSetupClient(
                name="Alice",
                model_constructor=model_constructor,
                account_idx=0,
                contract_address=None,
                deploy=True
            )
            trainers = self._make_trainers(
                dataset_constructor=dataset_constructor,
                model_constructor=model_constructor,
                model_criterion=model_criterion,
                split_type=split_type,
                num_trainers=num_trainers,
                ratios=ratios,
                flip_probs=flip_probs,
                disjointness=disjointness,
                unique_digits=unique_digits,
                client_constructor=ConsortiumClient,
                contract_address=alice.contract_address)
            alice.set_genesis_model(
                round_duration=self.ROUND_DURATION,
                max_num_updates=len(trainers)
            )
            alice.add_auxiliaries([
                trainer.address for trainer in trainers
            ])
        return alice, trainers

    def _make_trainers(
        self,
        dataset_constructor,
        model_constructor,
        model_criterion,
        split_type,
        num_trainers,
        ratios,
        flip_probs,
        disjointness,
        unique_digits,
        client_constructor,
        contract_address
    ):
        # instantiate data
        if split_type == 'equal' or split_type == 'dp':
            data, targets = dataset_constructor(
                train=True, subset=self.QUICK_RUN).split(num_trainers)
        if split_type == 'size':
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, ratios=ratios)
        if split_type == 'flip':
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, flip_probs=flip_probs)
        if split_type == 'noniid':
            data, targets = dataset_constructor(train=True, subset=self.QUICK_RUN).split(
                num_trainers, disjointness=disjointness)
        if split_type == 'unique_digits':
            assert dataset_constructor == MNISTData, "split_type=unique_digits only supported for MNISTData"
            dataset_unique = dataset_constructor(train=True, subset=self.QUICK_RUN,
                                                 exclude_digits=set(range(10))-set(unique_digits))
            data_unique = dataset_unique.data
            targets_unique = dataset_unique.targets
            data_others, targets_others = MNISTData(
                train=True, subset=self.QUICK_RUN, exclude_digits=unique_digits
            ).split(num_trainers-1)
            data = [data_unique] + data_others
            targets = [targets_unique] + targets_others

        # instantiate clients
        trainers = []
        names = ["Bob", "Carol", "David", "Eve",
                 "Frank", "Georgia", "Henry", "Isabel", "Joe"]
        for i, name, d, t in zip(range(num_trainers), names, data, targets):
            trainer = client_constructor(
                name=name,
                data=d,
                targets=t,
                account_idx=i+1,
                model_constructor=model_constructor,
                model_criterion=model_criterion,
                contract_address=contract_address,
                deploy=False
            )
            print(f"\t{name} counts: {torch.unique(t, return_counts=True)}")
            trainers.append(trainer)
        return trainers

    def _global_accuracy(self, client, one_hot_output):
        """
        A hacky function that gets the accuracy rather than the loss.
        """
        output = client.predict()
        if one_hot_output:
            pred = output.argmax(dim=2, keepdim=True).squeeze()
        else:
            pred = torch.round(output).squeeze()
        num_correct = (pred == client._targets).float().sum().item()
        accuracy = num_correct / len(pred)
        return accuracy

    def _label_counts(self, num_classes, trainers):
        digit_counts_by_name = {}
        for trainer in trainers:
            digit_counts = [0]*num_classes
            digits, counts = torch.unique(
                trainer._targets, return_counts=True)  # hacky
            digits = digits.int()
            for digit, count in zip(digits, counts):
                digit_counts[digit] = count.item()
            digit_counts_by_name[trainer.name] = digit_counts
        return digit_counts_by_name

    def _token_count_histories(self, trainers):
        token_count_history_by_name = {}
        for trainer in trainers:
            token_count_history_by_name[trainer.name] = [
                trainer.get_token_count(training_round=i)
                for i in range(1, self.TRAINING_ITERATIONS+1)
            ]
        return token_count_history_by_name

    def _gas_used(self, alice, trainers):
        gas_used_by_name = {}
        gas_used_by_name[alice.name] = alice.get_gas_used()
        for trainer in trainers:
            gas_used_by_name[trainer.name] = trainer.get_gas_used()
        return gas_used_by_name

    def _save_results(self, results):
        dataset = results['dataset']
        filedir = f"experiments/{dataset}/results/"
        if self.QUICK_RUN:
            filedir += "quick/"
        filename = "all.json"
        filepath = filedir + filename
        with open(filepath) as f:
            all_results = json.load(f)
        all_results.append(results)
        with open(filepath, 'w') as f:
            json.dump(all_results, f,
                      indent=4,
                      sort_keys=True)
        print(f"{filepath} now has {len(all_results)} results")



