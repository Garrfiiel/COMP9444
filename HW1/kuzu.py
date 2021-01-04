# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        output = self.linear(x.view(-1, 784))
        return F.log_softmax(output, dim=1)  # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.lin_hidden1 = torch.nn.Linear(784, 100)
        self.lin_output = torch.nn.Linear(100, 10)

    def forward(self, x):
        hidden_layer1 = torch.tanh(self.lin_hidden1(x.view(-1, 784)))
        output_layer = F.log_softmax(self.lin_output(hidden_layer1), dim=1)
        return output_layer  # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 48, kernel_size=4, stride=2, padding=2)  # 1+(28+2*2-4)/2=15
        self.conv2 = nn.Conv2d(48, 128, kernel_size=3, padding=1)  # 1+(15+2*1-3)/1=15
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1)  # 1+(15-3)/1=13
        self.lin1 = nn.Linear(128 * 13 * 13, 780)
        self.lin_output = nn.Linear(780, 10)
        self.relu = nn.ReLU()


    def forward(self, x):
        # print(x.size())
        conv_layer1 = self.relu(self.conv1(x))
        conv_layer2 = self.relu(self.conv2(conv_layer1))
        # print(conv_layer2.size())
        pool_layer = self.max_pool(conv_layer2)
        # print(pool_layer.size())
        lin_layer1 = self.relu(self.lin1(pool_layer.view(-1, 128 * 13 * 13)))
        output_layer = F.log_softmax(self.lin_output(lin_layer1), dim=1)
        return output_layer  # CHANGE CODE HERE
