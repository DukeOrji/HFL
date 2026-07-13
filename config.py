import torch
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

batch_size = 33
SAVE = False