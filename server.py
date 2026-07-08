#server.py
import torch.nn as nn
import torch.optim as optim
import torch
from user import norm
from config import device
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class Server:
    def __init__(self):
        self.global_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.global_model = self.global_model.to(device)
        self.global_model.fc = nn.Linear(512, 10).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        

    def broadcast_weight(self):
        return self.global_model.state_dict()

    def aggregate2(self, user_weights):
        



    def aggregate(self, user_weights):
        avg_weights = {}

        for key in user_weights[0].keys():
            if user_weights[0][key].dtype == torch.float32:
                weighted_sum = torch.zeros_like(user_weights[0][key])

                for idx, weights in enumerate(user_weights):
                    weighted_sum += weights[key]

                avg_weights[key] = weighted_sum / len(user_weights)

            else:
                # Copy non-trainable parameters directly
                avg_weights[key] = user_weights[0][key].clone()

        self.global_model.load_state_dict(avg_weights)
        
        return self.global_model.state_dict()
        

    

    def evaluate(self, dataloader):
        self.global_model.eval()
        losses = []

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:

                images = images.to(device)#send to gpu
                labels = labels.to(device)
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels).item()
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
                
                losses.append(loss)

            acc = round(correct/total, 2)
            avg_loss = round(sum(losses)/len(losses), 2)
        
        return avg_loss, acc
