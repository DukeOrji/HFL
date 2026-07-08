#main.py
from ds import load_cifar
import torch
from server import Server
from user import User
from config import device
import random
from torch.utils.data import random_split




num_clients = 20
user_dataloader, test_loader = load_cifar(num_clients)




users = [User(i, user_dataloader[i]) for i in range(num_clients)]


server = Server()



rng_num = 25
print(next(server.global_model.parameters()).device) #print gpu or cpu
for epoch in range(rng_num):
    print(f"\nRound: {epoch+1}")
    #server broadcast weight
    global_weight = server.broadcast_weight()

    for user in users:
        user.set_weight(global_weight)
        loss, acc = user.train()
        
        print(f"Client: {user.user_id}  |  Loss: {loss}  |  Acc: {acc}")

    user_weights = [
        user.get_weight()
        for user in users
    ]

    group_updates = random_split(
        user_weights,
        [4,4,4,4,4]
    )

    group_weigths = []
    for idx, weight in enumerate(group_updates):
        agg_upd = server.aggregate(weight)
        group_weigths.append(agg_upd)

    improved_w = []
    for i in range(len(group_weigths)):
        selected_w = random.sample(group_weigths, 3)
        improved_w.append(server.aggregate(selected_w))

    server.aggregate(random.sample(improved_w, 3))
    print("global weight set successfully")
    global_loss, global_acc = server.evaluate(test_loader)
    print(f"\nGlobal Loss: {global_loss}  |  Global Acc: {global_acc}")

   
    
    if epoch < rng_num-1:
        print("Next Round ...")
    else:
        print("Experiment Complete.")    

    
    


