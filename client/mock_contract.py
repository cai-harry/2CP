class MockContract:
    """
    Replicates the API of the smart contract and its essential functionality.
    """

    def __init__(self):

        # Hash of the latest global model to train from.
        self.latestHash = None
        
        # Model updates in the current training round.
        self.updates = []

    def recordContribution(self, model_hash):
        self.updates.append(model_hash)

    def getContributions(self):
        return self.updates

    def getLatestHash(self):
        return self.latestHash

    def setLatestHash(self, model_hash):
        """Sets new global model and resets list of updates"""
        self.latestHash = model_hash
        self.updates = []
