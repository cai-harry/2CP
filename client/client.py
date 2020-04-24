import ipfshttpclient
import torch
from torch import nn, optim
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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

class Client:
    def __init__(self, data_dir, contract):
        self._data_dir = data_dir
        self._data_idx = 0
        self._ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
        self._contract = contract
        
    def run_train(self):
        model = self._get_model()
        model = self._train_model(model)
        uploaded_hash = self._upload_model(model)
        self._record_contribution(uploaded_hash)
        
    def _get_model(self):
        latest_hash = self._contract.functions.getLatestHash().call()
        with self._ipfs_client as ipfs:
            model = ipfs.cat(latest_hash)
        return model
    
    def _train_model(self, model):
        data, label = data_dir[data_idx]
        optimizer = optim.SGD()
        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        return model
    
    def _upload_model(self, model):
        with self._ipfs_client as ipfs:
            res = ipfs.add_bytes(model)
        uploaded_hash = res['Hash']
        return uploaded_hash
    
    def _record_contribution(self, uploaded_hash):
        self._contract.functions.recordContribution(uploaded_hash).transact()
            
        
class Org:
    def __init_(self, contract):
        self._ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
        self._contract = contract
    
    def run(self):
        model_hashes = self._get_model_hashes()
        models = self._get_models(model_hashes)
        avg_model = self._avg_model(models)
        self._set_latest_model(avg_model)
        
    def _get_model_hashes(self):
        return self._contract.functions.getContributions().call()
        
    def _get_models(self, model_hashes):
        models = []
        with self._ipfs_client as ipfs:
            for model_hash in model_hashes:
                model = ipfs.cat(model_hash)
                models.append(model)
        return models
    
    def _avg_model(self, models):
        sum_weights = 0
        sum_biases = 0
        with torch.no_grad():
            for model in models:
                sum_weights += model.weight.data
                sum_biases += model.bias.data
            avg_weights = sum_weights / len(models)
            avg_biases = sum_biases / len(models)
        avg_model = Model()
        avg_model.weight.set_(avg_weights)
        avg_model.bias.set_(avg_biases)
        return avg_model
    
    def _set_latest_model(self, avg_model):
        with self._ipfs_client as ipfs:
            res = ipfs.add_bytes(avg_model)
        model_hash = res['Hash']
        self._contract.functions.setLatestModel(model_hash).transact()