# TODO: change `conv1` to only have a 1 channel input (i.e. Grayscale)
# TODO: Test if pooling or dropout performs better

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# class Net(nn.Module):
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
        
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 20, 5)
#         self.conv2 = nn.Conv2d(20, 7, 5)
#         self.conv3 = nn.Conv2d(7, 9, 5)
        
#         self.fc1 = nn.Linear(12996, 128)
#         self.fc2 = nn.Linear(128, 1)

#         self.test = nn.Linear(1, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(self.batch_size, -1)
#         #print(x.size())
#         x = torch.tanh(self.fc1(x))
#         x = F.relu(self.fc2(x))

#        return torch.squeeze(x)

class Net(nn.Module):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, 5)
        self.conv2 = nn.Conv2d(7, 20, 5)
        #self.conv3 = nn.Conv2d(30, 10, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(35280 , 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.test = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(self.batch_size, -1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.squeeze(x)

# class Net(nn.Module):
#     def __init__(self, batch_size):
#         self.batch_size = batch_size
        
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 7, 5)
#         self.conv2 = nn.Conv2d(7, 20, 5)
#         self.conv2 = nn.Conv2d(20, 5, 5)
        
#         self.fc1 = nn.Linear(17640, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(self.batch_size, -1)
#         #print(x.size())
#         x = torch.tanh(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         return torch.squeeze(x)
