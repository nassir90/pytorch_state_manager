from weights import Weights
from log import Log

class State():
    def __init__(self, weights_path: str, messages_path: str, records_path: str, epoch: int):
        self.weights_path = weights_path
        self.messages_path = messages_path
        self.records_path = records_path
        self.epoch = epoch
        
    def log(self):
        return Log(self.messages_path, self.records_path)

    def weights(self):
        return Weights(self.weights_path)
