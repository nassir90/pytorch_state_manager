import torch
import os

class Weights():
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.weights_dir = os.path.dirname(weights_path)

    def write(self, model):
        if not os.path.isdir(self.weights_dir):
            os.mkdir(self.weights_dir)
        torch.save(model.state_dict(), self.weights_path)

    def exists(self):
        return os.path.isfile(self.weights_path)
    
    def load(self, model, device):
        if self.exists():
            model.load_state_dict(torch.load(self.weights_path, map_location=device))
