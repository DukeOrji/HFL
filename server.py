#server.py
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from config import device
from user import norm


class Server:
    def __init__(self):
        
        self.global_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.global_model = self.global_model.to(device)
        self.global_model.classifier[-1] = nn.Linear(self.global_model.classifier[-1].in_features, 10).to(device) #align classifier with cifar classes
        self.loss_fn = nn.CrossEntropyLoss()
        print(self.global_model.classifier)

        
    def broadcast_weight(self):
        return self.global_model.state_dict()


    def aggregate(self, client_weights):

        avg_weights = {}

        for key in client_weights[0]:

            if client_weights[0][key].dtype.is_floating_point:

                avg = torch.zeros_like(client_weights[0][key])

                for weights in client_weights:
                    avg += weights[key]

                avg /= len(client_weights)

                avg_weights[key] = avg

            else:

                avg_weights[key] = client_weights[0][key].clone()

        self.global_model.load_state_dict(avg_weights)


    def set_weight(self, weight):

        self.global_model.load_state_dict(weight)

    def predict(self, images):
        self.global_model.eval()
        with torch.no_grad():
            pred = self.global_model(norm(images))
            pred_labels = pred.argmax(dim=1)
        return pred_labels

    def evaluate(self, dataloader):

        total = 0
        correct = 0

        losses = []

        with torch.no_grad():
            model.eval()

            for images, labels in dataloader:

                images = images.to(device)
                labels = labels.to(device)
                pred = self.global_model(norm(images))
                pred_labels = pred.argmax(dim=1)

                loss = self.loss_fn(pred, labels)
                losses.append(loss.item())
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = mean(losses).round(2)
        return avg_loss, acc
