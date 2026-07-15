#server.py
import torch
import torch.nn as nn
from collections import Counter
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from statistics import mean
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

    def new_method(self, client_weights):
        global_weights = self.global_model.state_dict()
        client_info = []

        for idx, weights in enumerate(client_weights):
            total_distance = 0
            update = {}

            for key in weights.keys():

                if weights[key].dtype.is_floating_point:

                    delta = weights[key] - global_weights[key]
                    update[key] = delta

                    total_distance += delta.norm().item()

                else:
                    update[key] = weights[key].clone()

            client_info.append({
                "idx": idx,
                "distance": total_distance,
                "update": update
            })

        client_info.sort(key=lambda x: x["distance"])

        selected = []

        selected.extend(client_info[:3])
        selected.append(client_info[len(client_info) // 2])
        selected.extend(client_info[-2:])

        avg_update = {}

        for key in global_weights.keys():

            if global_weights[key].dtype.is_floating_point:

                avg = torch.zeros_like(global_weights[key])

                for client in selected:
                    avg += client["update"][key]

                avg /= len(selected)

                avg_update[key] = avg

            else:
                avg_update[key] = global_weights[key].clone()

        new_weights = {}

        for key in global_weights.keys():

            if global_weights[key].dtype.is_floating_point:
                new_weights[key] = global_weights[key] + avg_update[key]

            else:
                new_weights[key] = global_weights[key].clone()

        self.global_model.load_state_dict(new_weights)




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

        #self.global_model.load_state_dict(avg_weights)
        return avg_weights


    def set_weight(self, weight):

        self.global_model.load_state_dict(weight)

    def predict(self, images):
        self.global_model.eval()
        with torch.no_grad():
            pred = self.global_model(norm(images))
            pred_labels = pred.argmax(dim=1)
        return pred_labels

    def ensemble(self, group_models, dataloader):
        correct = 0
        total = 0
        
        print("")
        self.global_model.eval()
        with torch.no_grad():

            for idx, (images, labels) in enumerate(dataloader):

                images = images.to(device)
                labels = labels.to(device)

                predictions = []
        
                for model in group_models:

                    self.set_weight(model)
                    pred = self.global_model(norm(images))
                    pred_labels = pred.argmax(dim=1)
                    predictions.append(pred_labels)

                majority_predictions = []

                for i in range(len(labels)):
                    img_votes = [pred[i].item() for pred in predictions]

                    majority = Counter(img_votes).most_common(1)[0][0]
                    majority_predictions.append(majority)

                majority_predictions = torch.tensor(majority_predictions, device=labels.device)
                correct += (majority_predictions == labels).sum().item()
                total += labels.size(0)

        return correct, total



    def evaluate(self, dataloader):

        total = 0
        correct = 0

        losses = []

        with torch.no_grad():
            self.global_model.eval()

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
