import json
import os
import re
import torch

def generate_full_pattern(end_pattern: str):
    return "(?!\.)(.*)" + end_pattern + "$"
    
class State():
    def __init__(self, epoch: int, suffix: str, weights_dir: str, messages_dir: str, records_dir: str, exists=False):
        self.exists = exists
        self.epoch = epoch
        self.suffix = suffix
        self.weights_dir = weights_dir
        self.records_dir = records_dir
        self.messages_dir = messages_dir
        self.weights_path = os.path.join(weights_dir, "%d%s" % (epoch, suffix))
        self.messages_path = os.path.join(messages_dir, "%d.txt" % epoch)
        self.records_path = os.path.join(records_dir, "%d.json" % epoch)

    def advanced(self):
        return State(self.epoch + 1, self.suffix, self.weights_dir, self.messages_dir, self.records_dir, self.exists)
    def log(self):
        return Log(self.messages_path, self.records_path)

    def stage(self, model):
        return Stage(model, self.weights_path)
        
    def give_weights(self, model, device):
        model.load_state_dict(torch.load(self.weights_path, map_location=device))

class Stage():
    def __init__(self, model: "torch.nn.Module", weights_path: str):
        self.model = model
        self.weights_path = weights_path

    def write(self):
        torch.save(self.model.state_dict(), self.weights_path)

class Log():
    def __init__(self,  messages_path: str, records_path: str):
        self.buf = ""
        self.records = {}
        self.messages_path = messages_path
        self.records_path = records_path
        
        if os.path.isfile(messages_path):
            with open(messages_path) as messages_file:
                self.buf = messages_file.read()
        if os.path.isfile(records_path):
            with open(records_path) as records_file:
                self.records = json.load(records_file)

    def log_and_print(self, *messages, end="\n"):
        self.log(*messages, end=end)
        print(*messages, end=end)

    def record_and_print(self, message: str, key: str, value, end="\n"):
        self.records[key] = value
        self.log_and_print(message % value, end=end)
    
    def log(self, *messages, end="\n"):
        for i, message in enumerate(messages):
            self.buf += str(message) + (" " if i != len(messages) - 1 else "")
        self.buf += end

    def write(self):
        if self.buf: 
            with open(self.messages_path, "w") as output_file:
                output_file.write(self.buf)
        if self.records:
            with open(self.records_path, "w") as output_file:
                json.dump(self.records, output_file)
                

class StateManager():
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
    
    def most_recent_state(self):
        epoch, suffix, exists = 0, ".weights", False
        for path in os.listdir(self.weights_dir):
            parts = re.match(generate_full_pattern(self.end_pattern), path)
            if parts and (not exists or int(parts.group(1)) > epoch):
                epoch, suffix, exists = int(parts.group(1)), parts.group(2), True
        return State(epoch, suffix, self.weights_dir, self.messages_dir, self.records_dir, exists)
    
    def stage_and_log(self, model: "torch.nn.Module"):
        new_state = self.most_recent_state().advanced()
        return new_state.stage(model), new_state.log()
