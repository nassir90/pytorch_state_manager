from pytorch_state_manager import StateManager as S, Log as L
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

s = S('weights', 'messages', 'records')
most_recent_state = s.most_recent_state()
most_recent_state.give_weights(model, device)
print('The previous epoch saw a training loss of :', most_recent_state.log().records['training_loss'])
g, l = s.stage_and_log(model)
l.log("hello", 5)
training_loss = 10
l.record_and_print("Training loss is %f", 'training_loss', training_loss)
l.write()
g.write()

print(s.most_recent_state().epoch)
