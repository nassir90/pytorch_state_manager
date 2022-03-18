import os
import re
import torch

def generate_full_pattern(end_pattern: str):
    return "(?!\.)(.*)" + end_pattern + "$"

class StateManager():
    def __init__(self, weights_dir, metadata_dir, end_pattern="(\.(pth?|weights))$"):
        if not os.path.isdir(weights_dir):
            os.mkdir(metadata_dir)
        self.weights_dir = weights_dir
        if not os.path.isdir(metadata_dir):
            os.mkdir(metadata_dir)
        self.metadata_dir = metadata_dir
        self.end_pattern = end_pattern

    def determine_most_recent_state(self):
        state = State(0, ".weights", exists=False)
        for path in os.listdir(self.weights_dir):
            parts = re.match(generate_full_pattern(self.end_pattern), path)
            if parts and (not state.exists or int(parts.group(1)) > state.epoch):
                state = State(epoch=int(parts.group(1)), suffix=parts.group(2), exists=True)
        return state
    
    def give_most_recent_weights(self, model: "torch.nn.Module", map_location: "torch.cuda.device"):
        last_state = self.determine_most_recent_state()
        if last_state.exists:
            model.load_state_dict(torch.load(os.path.join(self.weights_dir, last_state.get_weights_basename()), map_location=map_location))
        else:
            print("No saved models")
        
    def commit(self, model: "torch.nn.Module", message: str = ""):
        new_state = self.determine_most_recent_state().advanced()
        new_weights_path = os.path.join(self.weights_dir, new_state.get_weights_basename())
        torch.save(model.state_dict(), new_weights_path)
        if message:
            new_message_path = os.path.join(self.metadata_dir, new_state.get_message_basename())
            with open(new_message_path, "w") as message_file:
                message_file.write(message)

class State():
    def __init__(self, epoch: int, suffix: str, exists=True):
        self.epoch = epoch
        self.suffix = suffix
        self.exists = exists

    def get_weights_basename(self):
        return "%d%s" % (self.epoch, self.suffix)

    def get_message_basename(self):
        return "%d.txt" % self.epoch
        
    def advanced(self):
        return State(self.epoch + 1, self.suffix)
