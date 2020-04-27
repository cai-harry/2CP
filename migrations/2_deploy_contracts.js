var FederatedLearning = artifacts.require("./FederatedLearning.sol");

module.exports = async function(deployer) {
  await deployer.deploy(FederatedLearning);
  let crt = await FederatedLearning.deployed();
};