#user.py
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from config import device

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (
            x - self.mean.type_as(x)[None,:,None,None]
        ) / self.std.type_as(x)[None,:,None,None]
    
norm = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class User:
    def __init__(self, user_id, dataloader):
        self.user_id = user_id
        self.dataloader = dataloader
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model = self.model.to(device)
        self.model.fc = nn.Linear(512, 10).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-2)

    def train(self):
        self.model.train()
        correct = 0
        total = 0
        losses = []


        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx > 64:
                break

            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = self.model(norm(images))
            pred_labels = pred.argmax(dim=1)

            loss = self.loss_fn(pred, labels)

            #back propagation
            self.opt.zero_grad()
            loss.backward() 
            self.opt.step()
            losses.append(loss.item())

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(sum(losses)/len(losses), 2)

        return avg_loss, acc

    def get_weight(self):
        return self.model.state_dict()

    def set_weight(self, global_weight):

        self.model.load_state_dict(global_weight)

class SuperiorClient(User):


    def train(self, dataloader):
        self.model.train()
        correct = 0
        total = 0
        losses = []


        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx > 64:
                break
            
            images = images.to(device)#send to gpu
            labels = labels.to(device)

            pred = self.model(norm(images))
            pred_labels = pred.argmax(dim=1)

            loss = self.loss_fn(pred, labels)

            #back propagation
            self.opt.zero_grad()
            loss.backward() 
            self.opt.step()
            losses.append(loss.item())

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            acc = round(correct/total, 2)
            avg_loss = round(sum(losses)/len(losses), 2)

        return avg_loss, acc