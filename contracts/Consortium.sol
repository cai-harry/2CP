pragma solidity >=0.4.21 <0.7.0;

import "./Crowdsource.sol";


/// @title Deploys and manages Crowdsourcing auxiliaries that make up a Consortium Federated Learning process.
/// @author Harry Cai
contract Consortium {

    address internal mainAddress;

    address[] internal auxAddresses;

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

    /// @return The addresses of the Crowdsourcing auxiliaries.
    function auxiliaries() external view returns (address[] memory) {
        return auxAddresses;
    }

    /// @return count Token count of the given address.
    function countTokens(address _address) public view returns (uint256 count) {
        for (uint256 i = 0; i < auxAddresses.length; i++) {
            Crowdsource aux = Crowdsource(auxAddresses[i]);
            count += aux.countTokens(_address);
        }
    }

    /// @return count Token count of the calling address.
    function countTokens() external view returns (uint256 count) {
        count = countTokens(msg.sender);
    }

    /// @return count Total number of tokens.
    function countTotalTokens() external view returns (uint256 count) {
        for (uint256 i = 0; i < auxAddresses.length; i++) {
            Crowdsource aux = Crowdsource(auxAddresses[i]);
            count += aux.countTotalTokens();
        }
    }

    function setGenesis(bytes32 _modelHash, uint256 roundDuration) external {
        genesis = _modelHash;
        roundMinDuration = roundDuration;
        Crowdsource main = Crowdsource(mainAddress);
        main.setGenesis(genesis, roundMinDuration);
    }

    function addAux(address _evaluator) external {
        require(genesis != 0, "Genesis not set");
        Crowdsource aux = new Crowdsource();
        aux.setGenesis(genesis, roundMinDuration);
        aux.setEvaluator(_evaluator);
        auxAddresses.push(address(aux));
    }
}
