pragma solidity >=0.4.21 <0.7.0;

import "./Crowdsource.sol";


/// @title Deploys and manages Crowdsourcing subs that make up a Consortium Federated Learning process.
/// @author Harry Cai
contract Consortium {

    address internal mainAddress;

    address[] internal subsAddresses;

    bytes32 internal genesis;

    uint256 internal roundMinDuration;

    constructor() public {
        Crowdsource main = new Crowdsource();
        mainAddress = address(main);
    }

    /// @return The address of the Crowdsourcing main.
    function main() external view returns (address) {
        return mainAddress;
    }

    /// @return The addresses of the Crowdsourcing subs.
    function subs() external view returns (address[] memory) {
        return subsAddresses;
    }

    /// @return count Token count of the given address.
    function countTokens(address _address) public view returns (uint256 count) {
        for (uint256 i = 0; i < subsAddresses.length; i++) {
            Crowdsource sub = Crowdsource(subsAddresses[i]);
            count += sub.countTokens(_address);
        }
    }

    /// @return count Token count of the calling address.
    function countTokens() external view returns (uint256 count) {
        count = countTokens(msg.sender);
    }

    /// @return count Total number of tokens.
    function countTotalTokens() external view returns (uint256 count) {
        for (uint256 i = 0; i < subsAddresses.length; i++) {
            Crowdsource sub = Crowdsource(subsAddresses[i]);
            count += sub.countTotalTokens();
        }
    }

    function setGenesis(bytes32 _modelHash, uint256 roundDuration) external {
        genesis = _modelHash;
        roundMinDuration = roundDuration;
        Crowdsource main = Crowdsource(mainAddress);
        main.setGenesis(genesis, roundMinDuration);
    }

    function addSub(address _evaluator) external {
        require(genesis != 0, "Genesis not set");
        Crowdsource sub = new Crowdsource();
        sub.setGenesis(genesis, roundMinDuration);
        sub.setEvaluator(_evaluator);
        subsAddresses.push(address(sub));
    }
}
