#cert.py
from ds import load_cifar
import torch
import copy
from collections import Counter
from server import Server
from user import User
from config import device, SAVE
import random
import os
from attack import LabelPoison, SignFlip, WeightMan
import pandas as pd




num_attackers = 6
num_clients = 20
user_dataloader, test_loader = load_cifar(num_clients)

results = []

users = [User(i, user_dataloader[i]) for i in range(num_clients - num_attackers)]

users.extend(
    LabelPoison(i, user_dataloader[i])
    for i in range(num_clients - num_attackers, num_clients)
)

attack_type = {
    14: "LP",
    15: "LP",
    16: "LP",
    17: "LP",
    18: "LP",
    19: "LP"
}

random.shuffle(users)
server = Server()



rng_num = 40
print(next(server.global_model.parameters()).device) #print gpu or cpu
for epoch in range(rng_num):
    print(f"\nRound: {epoch+1}")
    #server broadcast weight
    global_weight = server.broadcast_weight()

    for user in users:
        user.set_weight(global_weight)
        loss, acc = user.train()
        
        username = attack_type.get(user.user_id, user.user_id)
        print(f"Client: {username}  |  Loss: {loss}  |  Acc: {acc}")

    user_weights = [
        user.get_weight()
        for user in users
    ]

    if epoch == 0:
        random.shuffle(user_weights)
    group_updates = [
        user_weights[0:5],
        user_weights[5:10],
        user_weights[10:15],
        user_weights[15:20]
    ]

    group_models = []

    for group in group_updates:

        server.aggregate(group)
        group_models.append(copy.deepcopy(server.broadcast_weight()))



    correct = 0
    total = 0
    
    print("")
    for idx, (images, labels) in enumerate(test_loader):
        

        images = images.to(device)
        labels = labels.to(device)
        predictions = []
        for model in group_models:

            server.set_weight(model)
            predictions.append(server.predict(images))

        majority_predictions = []
        for i in range(len(labels)):
            img_votes = [pred[i].item() for pred in predictions]

            majority = Counter(img_votes).most_common(1)[0][0]
            majority_predictions.append(majority)

        majority_predictions = torch.tensor(majority_predictions, device=labels.device)
        correct += (majority_predictions == labels).sum().item()
        total += labels.size(0)

    print(f"Ensemble Acc: %{(round(correct/total, 3) * 100)}")

    
    if epoch < rng_num-1:
        print("Next Round ...")
    else:
        print("Experiment Complete.")    

    
# if SAVE:

#     attack = ""
#     defense = ""
#     save_dir = f"FLCert_feedback/{defense}/{attack}"

#     os.makedirs(save_dir, exist_ok=True)

#     df = pd.DataFrame(results)
#     df.to_csv(
#         f"{save_dir}/results.csv",
#         index=False
#     )   
    


