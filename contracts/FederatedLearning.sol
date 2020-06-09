pragma solidity >=0.4.21 <0.7.0;


/// @title Records contributions made to a consortium Federated Learning process
/// @author Harry Cai
contract FederatedLearning {
    /// @notice Address of contract creator, who evaluates updates
    address public evaluator;

    /// @notice IPFS hash of genesis model
    bytes32 public genesis;

    mapping(uint256 => bytes32[]) private updatesInRound;

    mapping(address => bytes32[]) private updatesFromAddress;

    mapping(bytes32 => bool) private tokensAssigned;

    mapping(bytes32 => uint256) private tokens;

    uint256 internal genesisBlockNum;

    /// @notice Constructor. The address that deploys the contract is set as the evaluator.
    constructor() public {
        evaluator = msg.sender;
    }

    modifier evaluatorOnly() {
        require(msg.sender == evaluator, "Not the registered evaluator");
        _;
    }

    /// @return The index of the current training round.
    function currentRound() public view returns (uint256) {
        return 1 + block.number - genesisBlockNum;
    }

    /// @return The CID's of updates in the given training round.
    function updates(uint256 _round) external view returns (bytes32[] memory) {
        return updatesInRound[_round];
    }

    /// @return count Token count of the given address.
    function countTokens(address _address) public view returns (uint256 count) {
        bytes32[] memory updates = updatesFromAddress[_address];
        for (uint256 i = 0; i < updates.length; i++) {
            count += tokens[updates[i]];
        }
    }

    /// @return count Token count of the calling address.
    function countTokens() external view returns (uint256 count) {
        count = countTokens(msg.sender);
    }

    /// @return count Total number of tokens.
    function countTotalTokens() external view returns (uint256 count) {
        for (uint256 i = 1; i <= currentRound(); i++) {
            bytes32[] memory updates = updatesInRound[i];
            for (uint256 j = 0; j < updates.length; j++) {
                count += tokens[updates[j]];
            }
        }
    }

    /// @notice Starts training by setting the genesis model.
    /// @dev Does not reset the training process! Deploy a new contract instead.
    function setGenesis(bytes32 _modelHash) external evaluatorOnly() {
        genesis = _modelHash;
        genesisBlockNum = block.number;
    }

    /// @notice Records a training contribution in the current round.
    function addModelUpdate(bytes32 _cid, uint256 _round)
        external
    // trainersOnly()
    {
        require(_round > 0, "Trying to add an update for the genesis round");
        require(_round >= currentRound(), "Trying to add an update for a past round");
        require(_round <= currentRound(), "Trying to add an update for a future round");
        updatesInRound[_round].push(_cid);
        updatesFromAddress[msg.sender].push(_cid);
    }

    /// @notice Assigns a token count to an update.
    /// @param _cid The update being rewarded
    /// @param _numTokens The number of tokens to award; should be based on marginal value contribution
    function setTokens(bytes32 _cid, uint256 _numTokens)
        external
        evaluatorOnly()
    {
        require(
            !tokensAssigned[_cid],
            "Update has already been rewarded"
        );
        tokens[_cid] = _numTokens;
        tokensAssigned[_cid] = true;
    }
}
