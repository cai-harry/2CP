var Crowdsource = artifacts.require("./Crowdsource.sol");
var Consortium = artifacts.require("./Consortium.sol");

module.exports = async function(deployer) {
  let crowdsource = deployer.deploy(Crowdsource);
  let consortium = deployer.deploy(Consortium);
  await Promise.all([crowdsource, consortium]);
};