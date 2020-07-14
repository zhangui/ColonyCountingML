# TODO: Tune parameters, model, and loss function
# TODO: Check for bugs
# TODO: Check each image group has labels.txt
# TODO: Add img_dim as a possible shell argument
# TODO: Add batch_size as a possible shell argument

from model import Net
from preprocess import get_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

import sys
import os.path

import operator

# Arguments Format:
#     python.py test.py `network path` `data` 

# Validate parameters
num_of_param = len(sys.argv) - 1
if num_of_param < 2:
    print('Error: Must enter at least 2 arguments')
    exit()

# Default Values
epochs = 5
lr = .001
#batch_size = 25
batch_size = 1
momentum = .9

# Parse data
network_path = sys.argv[1]
data_path = sys.argv[2]


# Check if data path exists and its contents is correctly formatted
if not os.path.isdir(data_path):
    print('Error: Data path is not a directory')
    exit()


if len(os.listdir(data_path)) == 0:
    print('Error: Data path is empty')
    exit()


img_dim = (50, 50)

test_data, _, _  = get_loader(data_path, '.', img_dim, batch_size, shuffle=False, train=False)
#print(test_data.label)
#test_data[0]

# Training
net = Net(batch_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# If file exists then load
if os.path.isfile(network_path):
    print('Loading network at ' + network_path +  '...')
    net.load_state_dict(torch.load(network_path))

# And validate
loss = 0
for i, images in enumerate(test_data, 0):
    outputs = net(images)
    #print(i)
    #mean = torch.mean(num_of_colonies)
    #std = torch.std(num_of_colonies)
    #num_of_colonies = (num_of_colonies - mean) / std

    #print(i, torch.round(outputs * std + mean))
    mean = 49.905
    std = 28.6526
    print(i, torch.round(outputs * std + mean).item())
