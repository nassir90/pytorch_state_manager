from npsm import StateManager as S
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

s = S('weights', 'messages')
s.give_most_recent_weights(model, device)
s.commit(model)
s.commit(model)
