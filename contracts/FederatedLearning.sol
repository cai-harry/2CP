pragma solidity >=0.4.21 <0.7.0;

/// @title Records contributions made to a consortium Federated Learning process
/// @author Harry Cai
contract FederatedLearning {

    /// @notice Address of contract creator, who evaluates updates
    address public evaluator;

    /// @notice Addresses of registered trainers
    address[] public trainers;

    /// IPFS hash of genesis model
    bytes32 public genesis;

    /// @notice Mapping of trainer addresses to IPFS hashes of model updates in the previous training round. Clients will need to download all of these and train from their aggregate.
    /// @dev IPFS hashes are 34 bytes long but we discard the first two (0x1220), which indicate hash function and length. https://ethereum.stackexchange.com/questions/17094/how-to-store-ipfs-hash-using-bytes
    mapping(address => bytes32) public previousUpdates;

    /// @notice Mapping of trainer addresses IPFS hashes of model updates in the current training round.
    mapping(address => bytes32) public currentUpdates;

    /// @notice Index of training round. Starts at zero.
    uint256 public trainingRound;

    /// @notice Number of contributions each trainer has made
    mapping(address => uint256) public tokens;

    /// @notice Number of tokens issued and owned by trainers.
    uint256 public totalTokens;

    /// @notice Constructor. The address that deploys the contract is set as the evaluator.
    constructor() public {
        evaluator = msg.sender;
    }

    modifier trainersOnly() {
        require(isTrainer(msg.sender), "Not a registered trainer");
        _;
    }

    modifier evaluatorOnly() {
        require(msg.sender == evaluator, "Not the registered evaluator");
        _;
    }

    modifier endOfTrainingRoundOnly() {
        require(
            isTrainingRoundFinished(),
            "Training round is still in progress"
        );
        _;
    }

    /// @return Token count of the calling address.
    function getTokens() external view returns (uint256) {
        return tokens[msg.sender];
    }

    /// @return List of addresses of registered trainers.
    function getTrainers() external view returns (address[] memory) {
        return trainers;
    }

    /// @return Whether the calling address is a registered trainer.
    function isTrainer() external view returns (bool) {
        return isTrainer(msg.sender);
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

    /// @return Whether the current training round is finished.
    function isTrainingRoundFinished() public view returns (bool) {
        // check if currentUpdates maps to zero for any trainer
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            if (currentUpdates[trainer] == 0) {
                return false;
            }
        }
        return true;
    }

    /// @notice Starts or resets training and sets the genesis model.
    function setGenesis(bytes32 _modelHash) external evaluatorOnly() {
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            previousUpdates[trainer] = 0;
            currentUpdates[trainer] = 0;
            tokens[trainer] = 0;
        }
        delete trainers;
        totalTokens = 0;

        genesis = _modelHash;
        trainingRound = 0;
    }

    /// @notice Registers the calling address as a trainer.
    function addTrainer() external {
        require(!isTrainer(msg.sender), "Trainer already added");
        trainers.push(msg.sender);
    }

    /// @notice Records a training contribution in the current round.
    function addModelUpdate(bytes32 _modelHash) external trainersOnly() {
        require(
            currentUpdates[msg.sender] == 0,
            "Already submitted model update this round"
        );
        currentUpdates[msg.sender] = _modelHash;
    }

    /// @notice Rewards a trainer with tokens.
    /// @param _trainer The address of the trainer to give tokens to
    /// @param _numTokens The number of tokens to award; should be based on marginal value contribution
    function giveTokens(address _trainer, uint256 _numTokens)
        external
        evaluatorOnly()
        endOfTrainingRoundOnly()
    {
        require(isTrainer(_trainer), "Recipient is not a registered trainer");
        require(
            currentUpdates[_trainer] != 0,
            "Trainer did not submit a model update this round"
        );
        tokens[_trainer] = tokens[_trainer] + _numTokens;
        totalTokens = totalTokens + _numTokens;
    }

    /// @notice Ends the current training round and starts the next one.
    function nextTrainingRound()
        external
        evaluatorOnly()
        endOfTrainingRoundOnly()
    {
        // make previousUpdates equal to currentUpdates
        // reset currentUpdates so all trainers map to zero
        for (uint256 i = 0; i < trainers.length; i++) {
            address trainer = trainers[i];
            previousUpdates[trainer] = currentUpdates[trainer];
            currentUpdates[trainer] = 0;
        }
        trainingRound = trainingRound + 1;
    }
}
