import os
import re

from state import State

class UnifiedStateManager():
    def __init__(self, checkpoints_dir: str):
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir
    
    def last_state(self):
        epochs = self.epochs()
        return self.recall(max(epochs) if len(epochs) > 0 else 0)

    def epochs(self):
        return [int(path) for path in os.listdir(self.checkpoints_dir) if re.match("^[0-9]+$", path)]
    
    def recall(self, epoch: int):
        epoch_string = str(epoch)
        weights_path = os.path.join(self.checkpoints_dir, epoch_string, "weights.pt")
        messages_path = os.path.join(self.checkpoints_dir, epoch_string, "messages.txt")
        records_path = os.path.join(self.checkpoints_dir, epoch_string, "records.json")
        return State(weights_path, messages_path, records_path, epoch)
    
    def stage(self):
        new_state = self.recall(self.last_state().epoch + 1)
        return new_state.weights(), new_state.log()
