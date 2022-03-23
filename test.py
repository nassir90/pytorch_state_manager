from unified import UnifiedStateManager as U
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

s = U('checkpoints')
last_state = s.recall(1)
print(last_state.epoch, "does", "exist" if last_state.weights().exists() else "not exist")
last_state.weights().load(model, device)
w, l = s.stage()
l.log("hello", 5)
training_loss = 10
l.record_and_print("Training loss is %f", 'training_loss', training_loss)
l.write()
w.write(model)

print(s.last_state().epoch)
