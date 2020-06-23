import time

import functools
import methodtools

import syft as sy
import torch
from torch import nn, optim
from torch.nn import functional as F

from ipfs_client import IPFSClient
from contract_clients import CrowdsourceContractClient, ConsortiumContractClient
import shapley

_hook = sy.TorchHook(torch)


class _BaseClient:
    """
    Abstract base client containing common features of the clients for both the
    Crowdsource Protocol and the Consortium Protocol.
    """

    def __init__(self, name, model_constructor, contract_constructor, account_idx, contract_address=None):
        self.name = name
        self._model_constructor = model_constructor
        self._contract = contract_constructor(account_idx, contract_address)
        self._account_idx = account_idx
        self.address = self._contract.address

    def get_token_count(self):
        return self._contract.countTokens(), self._contract.countTotalTokens()

    def wait_for_txs(self, txs):
        self._print(f"Waiting for {len(txs)} transactions...")
        receipts = []
        if txs:
            for tx in txs:
                receipts.append(self._contract.wait_for_tx(tx))
            txs.clear()  # clears the list of pending transactions passed in as argument
        return receipts

    def _print(self, msg):
        print(f"{self.name}: {msg}")


class _GenesisClient(_BaseClient):
    """
    Extends upon base client with the ability to set the genesis model to start training.
    """

    def __init__(self, name, model_constructor, contract_constructor, account_idx, contract_address=None):
        super().__init__(name, model_constructor,
                         contract_constructor, account_idx, contract_address)
        self._ipfs_client = IPFSClient()

    def set_genesis_model(self, round_duration):
        """
        Create, upload and record the genesis model.
        """
        self._print("Setting genesis...")
        genesis_model = self._model_constructor()
        genesis_cid = self._upload_model(genesis_model)
        tx = self._contract.setGenesis(genesis_cid, round_duration)
        return tx

    def _upload_model(self, model):
        """Uploads the given model to IPFS."""
        uploaded_cid = self._ipfs_client.add_model(model)
        return uploaded_cid


class CrowdsourceClient(_GenesisClient):
    """
    Full client for the Crowdsource Protocol.
    """

    TOKENS_PER_UNIT_LOSS = 1e18  # number of wei per ether
    CURRENT_ROUND_POLL_INTERVAL = 0.1

    def __init__(self, name, data, targets, model_constructor, model_criterion, account_idx, contract_address=None):
        super().__init__(name,
                         model_constructor,
                         CrowdsourceContractClient,
                         account_idx,
                         contract_address)
        self.data_length = min(len(data), len(targets))

        self._worker = sy.VirtualWorker(_hook, id=name)
        self._criterion = model_criterion
        self._data = data.send(self._worker)
        self._targets = targets.send(self._worker)
        self._data_loader = torch.utils.data.DataLoader(
            sy.BaseDataset(self._data, self._targets),
            batch_size=len(data),
            shuffle=True
        )

    def is_evaluator(self):
        return self._contract.evaluator() == self._contract.address

    def get_current_global_model(self, return_current_training_round=False):
        """
        Calculate, or get from cache, the current global model.
        """
        current_training_round = self._contract.currentRound()
        current_global_model = self._get_global_model(current_training_round)
        if return_current_training_round:
            return current_global_model, current_training_round
        return current_global_model

    def run_training_round(self, epochs, learning_rate):
        """
        Run training using own data, upload and record the contribution.
        """
        model, training_round = self.get_current_global_model(
            return_current_training_round=True)
        self._print(f"Training model, round {training_round}...")
        model = self._train_model(model, epochs, learning_rate)
        uploaded_cid = self._upload_model(model)
        self._print(f"Adding model update...")
        tx = self._record_model(uploaded_cid, training_round)
        return tx

    def evaluate_updates(self):
        num_rounds = self._contract.currentRound()
        self._print(f"Starting evaluation over {num_rounds} rounds...")
        scores = {}
        for r in range(1, num_rounds+1):
            scores.update(
                self._evaluate_updates(r)
            )
        return scores

    def set_tokens(self, cid_scores):
        """
        Record the given Shapley value scores for the given contributions.
        """
        txs = []
        self._print(f"Setting {len(cid_scores.values())} scores...")
        for cid, score in cid_scores.items():
            num_tokens = max(0, int(score * self.TOKENS_PER_UNIT_LOSS))
            tx = self._contract.setTokens(cid, num_tokens)
            txs.append(tx)
        return txs

    def evaluate_current_global(self):
        """
        Evaluate the current global model using own data.
        """
        current_training_round = self._contract.currentRound()
        return self._evaluate_global(current_training_round)

    def predict(self):
        model = self.get_current_global_model()
        model = model.send(self._worker)
        predictions = []
        with torch.no_grad():
            for data, labels in self._data_loader:
                data, labels = data.float(), labels.float()
                pred = model(data).get()
                predictions.append(pred)
        return torch.stack(predictions)

    def wait_for_round(self, n):
        self._print(f"Waiting for round {n}...")
        while(self._contract.currentRound() < n):
            time.sleep(self.CURRENT_ROUND_POLL_INTERVAL)

    @methodtools.lru_cache()
    def _get_global_model(self, training_round):
        """
        Calculate global model at the the given training round by aggregating updates from previous round.

        This can only be done if training_round matches the current contract training round.
        """
        model_cids = self._get_cids(training_round - 1)
        models = self._get_models(model_cids)
        avg_model = self._avg_model(models)
        return avg_model

    @methodtools.lru_cache()
    def _evaluate_global(self, training_round):
        """
        Evaluate the global model at the given training round.
        """
        model = self._get_global_model(training_round)
        loss = self._evaluate_model(model)
        return loss

    def _train_model(self, model, epochs, lr):
        model = model.send(self._worker)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epochs):
            for data, labels in self._data_loader:
                optimizer.zero_grad()
                pred = model(data)
                loss = self._criterion(pred, labels)
                loss.backward()
                optimizer.step()
        return model.get()

    def _evaluate_model(self, model):
        model = model.send(self._worker)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for data, labels in self._data_loader:
                pred = model(data)
                total_loss += self._criterion(pred, labels)
        avg_loss = total_loss / len(self._data_loader)
        model = model.get()
        return avg_loss.get().item()

    def _record_model(self, uploaded_cid, training_round):
        """Records the given model IPFS cid on the smart contract."""
        return self._contract.addModelUpdate(uploaded_cid, training_round)

    def _get_cids(self, training_round):
        if training_round < 0 or training_round > self._contract.currentRound():
            raise ValueError(
                f"training_round={training_round} out of bounds [0, {self._contract.currentRound()}]")
        if training_round == 0:
            return [self._contract.genesis()]
        cids = self._contract.updates(training_round)
        if not cids:  # if cids is empty, refer to previous round
            return self._get_cids(training_round - 1)
        return cids

    def _get_models(self, model_cids):
        models = []
        for cid in model_cids:
            model = self._ipfs_client.get_model(cid, self._model_constructor)
            models.append(model)
        return models

    def _get_genesis_model(self):
        gen_cid = self._contract.genesis()
        return self._ipfs_client.get_model(gen_cid, self._model_constructor)

    def _avg_model(self, models):
        avg_model = self._model_constructor()
        with torch.no_grad():
            for params in avg_model.parameters():
                params *= 0
            for client_model in models:
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    avg_param += client_param / len(models)
        return avg_model

    def _evaluate_updates(self, training_round):
        """
        Provide Shapley Value score for each update in the given training round.
        """
        self._print(f"Evaluating updates in round {training_round}...")

        cids = self._get_cids(training_round)

        def characteristic_function(*c):
            return self._marginal_value(training_round, *c)
        scores = shapley.values(
            characteristic_function, cids)

        self._print(f"Scores in round {training_round} are {list(scores.values())}")
        return scores

    @functools.lru_cache()
    def _marginal_value(self, training_round, *update_cids):
        """
        The characteristic function used to calculate Shapley Value.
        The Shapley Value of a coalition of trainers is the marginal loss reduction
        of the average of their models
        """
        start_loss = self._evaluate_global(training_round)
        models = self._get_models(update_cids)
        avg_model = self._avg_model(models)
        loss = self._evaluate_model(avg_model)
        return start_loss - loss


class ConsortiumSetupClient(_GenesisClient):
    """
    Client which sets up the Consortium Protocol but does not participate.
    """

    def __init__(self, name, model_constructor, account_idx, contract_address=None):
        super().__init__(name,
                         model_constructor,
                         ConsortiumContractClient,
                         account_idx,
                         contract_address)

    def add_sub(self, evaluator):
        self._print(f"Setting sub...")
        return self._contract.addSub(evaluator)


class ConsortiumClient(_BaseClient):
    """
    Full client for the Consortium Protocol.
    """

    def __init__(self, name, data, targets, model_constructor, model_criterion, account_idx, contract_address=None):
        super().__init__(name,
                         model_constructor,
                         ConsortiumContractClient,
                         account_idx,
                         contract_address)
        self._data = data
        self._targets = targets
        self._criterion = model_criterion
        self._main_client = CrowdsourceClient(name + " (main)",
                                              data,
                                              targets,
                                              model_constructor,
                                              model_criterion,
                                              account_idx,
                                              contract_address=self._contract.main())
        self._sub_clients = {}  # cache, updated every time self._get_sub_clients() is called

    def get_current_global_model(self):
        return self._main_client.get_current_global_model()

    def run_training_round(self, epochs, learning_rate):
        train_clients = self._get_train_clients()
        return [
            client.run_training_round(epochs, learning_rate)
            for client in train_clients
        ]

    def evaluate_updates(self):
        eval_clients = self._get_eval_clients()
        return [
            client.evaluate_updates()
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

    def wait_for_round(self, n):
        self._main_client.wait_for_round(n)
        for client in self._get_sub_clients():
            client.wait_for_round(n)

    def _get_sub_clients(self):
        """
        Updates self._sub_clients cache then returns it
        """
        for sub in self._contract.subs():
            if sub not in self._sub_clients.keys():
                self._sub_clients[sub] = CrowdsourceClient(
                    self.name + f" (sub {len(self._sub_clients)+1})",
                    self._data,
                    self._targets,
                    self._model_constructor,
                    self._criterion,
                    self._account_idx,
                    contract_address=sub
                )
        # No need to check to remove sub clients as the contract does not allow it
        return self._sub_clients.values()

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
