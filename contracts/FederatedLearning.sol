pragma solidity >=0.4.21 <0.7.0;

contract FederatedLearning {

    bytes latestHash;
    mapping(address=>bytes) public updates;

    function recordContribution(bytes memory _hash) public {
        updates[msg.sender] = _hash;
    }
    
    function getContributions() public view returns bytes[]{
        return updates;
    }

    function getLatestHash() public view returns bytes {
        return latestHash;
    }

    function setLatestHash(bytes memory _hash) public {
        latestHash = _hash;
    }
}
