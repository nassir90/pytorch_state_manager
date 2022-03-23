import os
import re

from state import State

def generate_full_pattern(end_pattern: str):
    return "(?!\.)(.*)" + end_pattern + "$"
                
class DistributedStateManager():
    def __init__(self, weights_dir, messages_dir, records_dir="", end_pattern="(\.(pth?|weights))$"):
        if not os.path.isdir(weights_dir):
            os.mkdir(metadata_dir)
        self.weights_dir = weights_dir
        if not os.path.isdir(messages_dir):
            os.mkdir(messages_dir)
        self.messages_dir = messages_dir
        if not records_dir:
            self.records_dir = messages_dir
        else:
            if not os.path.isdir(records_dir):
                os.mkdir(records_dir)
            self.records_dir = records_dir
        self.end_pattern = end_pattern
                          
    def last_state(self):
        epochs_and_suffixes = self.epochs_and_suffixes()
        if len(epochs_and_suffixes) > 0:
            epoch, suffix = epochs_and_suffixes[-1]
            return self.recall(epoch, suffix)
        else:
            return self.recall(0, ".weights")

    def recall(self, epoch, suffix=".weights"):
        weights_path = os.path.join(self.weights_dir, "%d%s" % (epoch, suffix))
        messages_path = os.path.join(self.messages_dir, "%d.txt" % epoch)
        records_path = os.path.join(self.records_dir, "%d.json" % epoch)
        return State(weights_path, messages_path, records_path, epoch)

    def epochs_and_suffixes(self):
        epochs_and_suffixes = []
        for path in os.listdir(self.weights_dir):
            parts = re.match(generate_full_pattern(self.end_pattern), path)
            if parts:
                epoch = int(parts.group(1))
                suffix = parts.group(2)
                epochs_and_suffixes.append((epoch, suffix))
        return sorted(epochs_and_suffixes)
    
    def stage(self):
        new_state = self.recall(self.last_state().epoch + 1)
        return new_state.weights(), new_state.log()
