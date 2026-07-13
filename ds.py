from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

conversion = transforms.Compose([
    transforms.ToTensor()
])

# Non-IID partitioning
def dirichlet_partition(dataset, num_clients, alpha, batch_size=64):
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    #store indices assigned to each client
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        #get all samples of class c
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        #sample class proportions
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        #convert proportions to sample counts
        proportions  = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        split_indices = np.split(class_indices, proportions)
        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx.tolist())

    #shuffle each clients data
    client_loaders = []

    for indices in client_indices:
        np.random.shuffle(indices)
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
        )

        client_loaders.append(loader)
    return client_loaders



def load_cifar(num_clients):
    #train dataset
    cifar_train = datasets.CIFAR10(
        root= "./data",
        train=True,
        download= True,
        transform= conversion
    )

    small_dataset = Subset(
        cifar_train,
        range(30000)
    )

    ud = len(small_dataset)//num_clients
    user_dataset = random_split(
        small_dataset,
        [ud for _ in range(num_clients)]
    )

    client_loaders = dirichlet_partition(
    cifar_train,
    num_clients=10,
    alpha=0.5
    )

    user_dataloader = [DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    ) for dataset in user_dataset]


    #test dataset
    cifar_test = datasets.CIFAR10(
        root= "./data",
        train=False,
        download= True,
        transform= conversion
    )

    test_loader = DataLoader(
        cifar_test,
        batch_size=64,
        shuffle=False
    )

    return user_dataloader, test_loader