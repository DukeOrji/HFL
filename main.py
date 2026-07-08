#main.py
from ds import load_cifar
import torch
from server import Server
from user import User, SuperiorClient
from config import device
from torch.utils.data import random_split




num_clients = 20
user_dataloader, test_loader, sup_dataloader = load_cifar(num_clients)




users = [User(i, user_dataloader[i]) for i in range(num_clients)]
sup_clients = [SuperiorClient(i, sup_dataloader[i]) for i in range(4)]

server = Server()



rng_num = 5
print(next(server.global_model.parameters()).device) #print gpu or cpu
for epoch in range(rng_num):
    print(f"\nRound: {epoch+1}")
    #server broadcast weight
    global_weight = server.broadcast_weight()

    #clients train on global weigth
    for user in users:
        user.set_weight(global_weight)
        loss, acc = user.train()
        print(f"Client: {user.user_id}  |  Loss: {loss}  |  Acc: {acc}")


    user_weights = [
        user.get_weight()
        for user in users
    ]

    #split updates into 4 groups
    group_updates = random_split(
        user_weights,
        [5, 5, 5, 5]
    )
    print("\nClient groups initialized...")

    group_weigths = []
    #aggregate group updates and store
    for idx, weights in enumerate(group_updates):
        agg_update = server.aggregate(weights)
        group_weigths.append(agg_update)
    print("group aggregation complete...\n")    

    improved_client_w = []

    #retrain on group weights to improve gradients
    for idx, weight in enumerate(group_weigths):
        
        sup_clients[idx].set_weight(weight)
        loss, acc = sup_clients[idx].train(sup_dataloader[idx])

        print(f"SuperiorClient: {idx}  |  Loss: {loss}  |  Acc: {acc}")
        
        improved_w = sup_clients[idx].get_weight()
        improved_client_w.append(improved_w)

    #aggregate improved weights and pass to global model
    final_weights = server.aggregate(improved_client_w)
    server.set_weight(final_weights)

    global_loss, global_acc = server.evaluate(test_loader)
    print(f"\nGlobal Loss: {global_loss}  |  Global Acc: {global_acc}")


    new_global_weights = server.broadcast_weight()
    for user in users:
        user.set_weight(new_global_weights)
    
    if epoch < rng_num-1:
        print("Next Round ...")
    else:
        print("Experiment Complete.")    

    
    


