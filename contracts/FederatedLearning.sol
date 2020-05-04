pragma solidity >=0.4.21 <0.7.0;


contract FederatedLearning {
    // Addresses of registered trainers
    address[] public trainers;

    // IPFS hashes of model updates in the previous training round.
    // Clients will need to download all of these and train from their aggregate.
    // IPFS hashes are 34 bytes long but we discard the first two (0x1220), which indicate hash function and length.
    // https://ethereum.stackexchange.com/questions/17094/how-to-store-ipfs-hash-using-bytes
    bytes32[] public previousUpdates;

    // IPFS hashes of model updates in the current training round.
    mapping(address => bytes32) public currentUpdates;

    // Number of contributions each trainer has made
    mapping(address => uint) public tokens;

    // Number of tokens issued and owned by trainers.
    uint public totalTokens;

    modifier trainersOnly() {
        require(isTrainer(msg.sender), "Not a registered trainer");
        _;
    }

    function getTokens() external view returns (uint) {
        return tokens[msg.sender];
    }

    function getPreviousUpdates() external view returns (bytes32[] memory) {
        return previousUpdates;
    }

    function setGenesis(bytes32 _modelHash) external {
        delete previousUpdates;
        previousUpdates.push(_modelHash);
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            currentUpdates[trainer] = 0;
            tokens[trainer] = 0;
        }
        delete trainers;
        totalTokens = 0;
    }

    function addTrainer() external {
        require(!isTrainer(msg.sender), "Trainer already added");
        trainers.push(msg.sender);
    }

    function addModelUpdate(bytes32 _modelHash) external trainersOnly() {
        require(currentUpdates[msg.sender] == 0, "Already submitted model update this round");
        currentUpdates[msg.sender] = _modelHash;
        if (isTrainingRoundFinished()) {
            nextTrainingRound();
        }
        tokens[msg.sender] = tokens[msg.sender] + 1;
        totalTokens = totalTokens + 1;
    }

    function isTrainer(address a) internal view returns (bool) {
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            if (trainer == a) {
                return true;
            }
        }
        return false;
    }

    function isTrainingRoundFinished() internal view returns (bool) {
        // check if currentUpdates maps to zero for any trainer
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            if (currentUpdates[trainer] == 0) {
                return false;
            }
        }
        return true;
    }

    function nextTrainingRound() internal {
        delete previousUpdates;

        // reset currentUpdates so all trainers map to zero
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            previousUpdates.push(currentUpdates[trainer]);
            currentUpdates[trainer] = 0;
        }
    }
}
