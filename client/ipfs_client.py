import io

import ipfshttpclient
import torch

class IPFSClient:

    # class attribute so that IPFSClient's on the same machine can all benefit
    _cached_models = {}

    def __init__(self, model_constructor):
        self._ipfs_client = ipfshttpclient.connect(
            '/ip4/127.0.0.1/tcp/5001/http')
        self._model_constructor = model_constructor

    def get_model(self, model_cid):
        if model_cid in self._cached_models:
            return self._cached_models[model_cid]
        model = self._model_constructor()
        with self._ipfs_client as ipfs:
            model_bytes = ipfs.cat(model_cid)
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer))
        return model

    def add_model(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        with self._ipfs_client as ipfs:
            model_cid = ipfs.add_bytes(buffer.read())
        self._cached_models[model_cid] = model
        return model_cid

