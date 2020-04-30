pragma solidity >=0.4.21 <0.7.0;

contract FederatedLearning {

    // Number of contributions per round
    // TODO: time is probably a better thing to use
    uint public contributionsPerRound;

    // IPFS hashes of model updates in the current training round.
    // IPFS hashes are 34 bytes long but we discard the first two (0x1220), which indicate hash function and length.
    // https://ethereum.stackexchange.com/questions/17094/how-to-store-ipfs-hash-using-bytes
    bytes32[] public currentUpdates;

    // IPFS hashes of model updates in the previous training round.
    // Clients will need to download all of these and train from their aggregate.
    bytes32[] public previousUpdates;

    function getPreviousUpdates() external view returns (bytes32[] memory) {
        return previousUpdates;
    }

    function setGenesis(bytes32 _modelHash) external {
        require(previousUpdates.length == 0, "Training history is not empty");
        previousUpdates.push(_modelHash);
    }

    function addClient() external {
        contributionsPerRound = contributionsPerRound + 1;
    }

    function addModelUpdate(bytes32 _modelHash) external {
        currentUpdates.push(_modelHash);
        if (currentUpdates.length >= contributionsPerRound) {
            nextTrainingRound();
        }
    }

    function nextTrainingRound() internal {
        previousUpdates = currentUpdates;
        delete currentUpdates;
    }
}
