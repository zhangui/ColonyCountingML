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
#     python.py train.py `network path` `data` [epochs] [learning rate] [batch size] [momentum]
# This command will load, if it exists, an exists neural network
# from `network path`, and create it otherwise. It will train from data in the folder
# `data` (there should be two folders in here: train and validation). Optionally,
# parameters such as epochs, momentum, and learning rate can be given.
# Any other parameters entered are ignored.

# Validate parameters
num_of_param = len(sys.argv) - 1
if num_of_param < 2:
    print('Error: Must enter at least 2 arguments')
    exit()

# Default Values
epochs = 5
lr = .001
batch_size = 25
momentum = .9

# Parse data
network_path = sys.argv[1]
data_path = os.path.join(sys.argv[2], 'data')

if 2 < num_of_param:
    epochs = int(sys.argv[3])

if 3 < num_of_param:
    lr = float(sys.argv[4])
    
if 4 < num_of_param:
    batch_size = int(sys.argv[5])

if 5 < num_of_param:
    momentum = float(sys.argv[6])

print('Epochs: {epochs}'.format(epochs = epochs))
print('Learning Rate: {lr}'.format(lr = lr))
print('Batch Size: {batch_size}'.format(batch_size = batch_size))
print('Momentum: {momentum}'.format(momentum = momentum))
print(' ')

# Check if data path exists and its contents is correctly formatted
if not os.path.isdir(data_path):
    print('Error: Data path is not a directory')
    exit()

def check_subdir(path, subdir):
    data_path_contents = os.listdir(data_path)
    if subdir  not in data_path_contents:
        print('Error: Data path has no subdirectory "' + subdir + '"')
        exit()

    if len(os.listdir(os.path.join(data_path, subdir))) == 0:
        print('Error: Data path\'s subdirectory "' + subdir + '" is empty')
        exit()

    if not os.path.isfile(os.path.join(data_path, subdir, 'labels.csv')):
        print('Error: Data path\'s subdirectory "' + subdir + '" has no labels file')
        exit()

    
check_subdir(data_path, 'train')
check_subdir(data_path, 'validate')

img_dim = (50, 50)

train_data, mean, std = get_loader(data_path, 'train', img_dim, batch_size, shuffle=True)
test_data, _, _ = get_loader(data_path, 'validate', img_dim, batch_size, shuffle=False)

mean = 49.905
std = 28.6526

print('Mean', mean)
print('Std', std)

# Training
net = Net(batch_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# If file exists then load
if os.path.isfile(network_path):
    print('Loading network at ' + network_path +  '...')
    net.load_state_dict(torch.load(network_path))

loss_sum = 0

# Then we train
print('Training network...')
for epoch in range(epochs):
    for i, data in enumerate(train_data, 0):
        images, num_of_colonies = data
        
        # --- TESTING ---
        # Normalize
        num_of_colonies = (num_of_colonies - mean) / std
        # --- END TESTING ---

        optimizer.zero_grad() 
        outputs = net(images).double()
        loss = criterion(outputs, num_of_colonies)
        loss.backward()
        optimizer.step()

        loss_sum += loss
    #print('epochs', epoch)
    print(loss_sum.item())
    loss_sum = 0
        



# Save
torch.save(net.state_dict(), network_path)

print('here')
# And validate
loss = 0
for i, data in enumerate(test_data, 0):
    images, num_of_colonies = data
            
    # --- TESTING ---
    # Normalize
    num_of_colonies = (num_of_colonies - mean) / std
    # --- END TESTING ---

    optimizer.zero_grad() 
    outputs = net(images)
    print(i, torch.round(outputs * std + mean))
    print(i, num_of_colonies * std + mean)
    loss += criterion(outputs, num_of_colonies)
