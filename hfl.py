#main.py
from ds import load_cifar
import torch
from server import Server
from user import User
from config import device, SAVE
import random
import os



num_clients = 20
user_dataloader, test_loader = load_cifar(num_clients)




users = [User(i, user_dataloader[i]) for i in range(num_clients)]


server = Server()
round_metrics = []
results = []


rng_num = 25
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


    random.shuffle(user_weights)

    group_updates = [
        user_weights[0:5],
        user_weights[5:10],
        user_weights[10:15],
        user_weights[15:20],
    ]
    print("\nClient groups initialized...")


    group_weights = []
    #aggregate group updates and store
    for idx, weights in enumerate(group_updates):
        agg_update = server.aggregate(weights)
        group_weights.append(agg_update)
    print("group aggregation complete...\n")    


    #aggregate group weights and pass to global model
    final_weight = server.aggregate(random.sample(group_weights))
    server.set_weight(final_weight)
    print("global weight set successfully")

    global_loss, global_acc = server.evaluate(test_loader)
    print(f"\nGlobal Loss: {global_loss}  |  Global Acc: {global_acc}")

    
    if epoch < rng_num-1:
        print("Next Round ...")
    else:
        print("Experiment Complete.")    


if SAVE:

    attack = ""
    defense = ""
    save_dir = f"feedback/{defense}/{attack}"

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(
        f"{save_dir}/results.csv",
        index=False
    )   
    


