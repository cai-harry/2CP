import os

import ipfshttpclient
import torch

class IPFSClient:
    def __init__(self):
        self._ipfs_client = ipfshttpclient.connect(
            '/ip4/127.0.0.1/tcp/5001/http')

    def get_model(self, model_cid, model_constructor):
        model = model_constructor()
        with self._ipfs_client as ipfs:
            # saves to current directory, filename is model_cid
            ipfs.get(model_cid)
        model.load_state_dict(torch.load(model_cid))
        os.remove(model_cid)
        return model

    def add_model(self, model):
        model_filename = "tmp.pt"
        with self._ipfs_client as ipfs:
            torch.save(model.state_dict(), model_filename)
            res = ipfs.add(model_filename)
        os.remove(model_filename)
        model_cid = res['Hash']
        return model_cid

