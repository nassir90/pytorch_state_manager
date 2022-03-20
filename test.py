from pytorch_state_manager import StateManager as S, Log as L
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

s = S('weights', 'messages', 'records')
last_state = s.last_state()
print(last_state.epoch)
last_state.weights().load(model, device)
w, l = s.stage()
l.log("hello", 5)
training_loss = 10
l.record_and_print("Training loss is %f", 'training_loss', training_loss)
l.commit()
w.commit(model)

print(s.most_recent_state().epoch)
