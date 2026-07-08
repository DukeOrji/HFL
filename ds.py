from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

conversion = transforms.Compose([
    transforms.ToTensor()
])

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

    user_dataloader = [DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    ) for dataset in user_dataset]


    #superior dataset to facilitate secondhand training
    sd = len(small_dataset)//4
    sup_dataset = random_split(
        small_dataset,
        [sd for _ in range(4)]
    )
    sup_dataloader = [DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    ) for dataset in sup_dataset]

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

    return user_dataloader, test_loader, sup_dataloader