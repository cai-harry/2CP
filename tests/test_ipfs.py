from ipfs_client import IPFSClient

from test_utils.xor import XORModel
from test_utils.functions import same_weights

def test_ipfs():
    ipfs = IPFSClient(XORModel)
    model = XORModel()
    cid = ipfs.add_model(model)
    assert cid[:2] == "Qm"
    downloaded_model = ipfs.get_model(cid)
    assert same_weights(model, downloaded_model)
