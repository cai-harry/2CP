pragma solidity >=0.4.21 <0.7.0;

contract FederatedLearning {

    // IPFS hash of the latest global model as a bytes32.
    // IPFS hashes are 34 bytes long but we discard the first two (0x1220), which indicate hash function and length.
    // https://ethereum.stackexchange.com/questions/17094/how-to-store-ipfs-hash-using-bytes
    bytes32 latestHash;

    // IPFS hashes of model updates in the current training round.
    bytes32[] updates;

    function recordContribution(bytes32 _hash) public {
        updates.push(_hash);
    }

    // TODO: do we need these getters?
    function getContributions() public view returns (bytes32[] memory) {
        return updates;
    }

    function getLatestHash() public view returns (bytes32) {
        return latestHash;
    }

    function setLatestHash(bytes32 _hash) public {
        latestHash = _hash;
        delete updates;
    }
}
