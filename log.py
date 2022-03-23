import json
import os

class Log():
    def __init__(self,  messages_path: str, records_path: str):
        self.buf = ""
        self.records = {}
        self.messages_path = messages_path
        self.messages_dir = os.path.dirname(messages_path)
        self.records_path = records_path
        self.records_dir = os.path.dirname(records_path)
        
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
        self.log_and_print(message.format(value), end=end)
    
    def log(self, *messages, end="\n"):
        for i, message in enumerate(messages):
            self.buf += str(message) + (" " if i != len(messages) - 1 else "")
        self.buf += end

    def write(self):
        
        if self.buf:
            if not os.path.isdir(self.messages_dir):
                os.mkdir(self.messages_dir)
            with open(self.messages_path, "w") as output_file:
                output_file.write(self.buf)
        if self.records:
            if not os.path.isdir(self.records_dir):
                self.mkdir(self.records_dir)
            with open(self.records_path, "w") as output_file:
                json.dump(self.records, output_file)
