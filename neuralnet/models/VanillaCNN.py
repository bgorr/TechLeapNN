import math
import os
import matplotlib
import netCDF4
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

import pickle
import warnings










class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # The purpose of pooling layers is to mitigate the sensitivity of convolutional layers to location and of spatially down-sampling representations
        # MaxPool2d - Max pooling calculates the maximum value of the elements in the pooling window
        self.mp = nn.MaxPool2d(2)



        self.fc = nn.Linear(8192, 2)
        # self.fc = nn.Linear(33280, 2)


    """
        --> FORWARD PROPAGATION: torch.Size([64, 7, 161, 105]) 64
        1 torch.Size([64, 32, 161, 105])
        2 torch.Size([64, 32, 80, 52])
        3 torch.Size([64, 64, 80, 52])
        4 torch.Size([64, 64, 40, 26])
        5 torch.Size([64, 128, 40, 26])
        6 torch.Size([64, 128, 20, 13])
        --> SHAPE BEFORE LINEAR TRANSFORM: torch.Size([64, 33280])
    
    
    
    
        --> FORWARD PROPAGATION: torch.Size([64, 7, 64, 64]) 64
        1 torch.Size([64, 32, 64, 64])
        2 torch.Size([64, 32, 32, 32])
        3 torch.Size([64, 64, 32, 32])
        4 torch.Size([64, 64, 16, 16])
        5 torch.Size([64, 128, 16, 16])
        6 torch.Size([64, 128, 8, 8])
        --> SHAPE BEFORE LINEAR TRANSFORM: torch.Size([64, 8192]) 
    """



    def forward(self, x):
        # Activation Functions
        #   - Rectified Linear Unit: ReLU(x) = max(x,0)
        #       Properties
        #       - only retains positive elements
        #       - piecewise linear
        #       - bounds: [0, inf)
        #       - if input positive, derivative equals 1
        #       - if input negative, derivative equals 0
        #   - Sigmoid Function: 1 / (1 + exp(-x))
        #       Properties
        #       - bounds: (0, 1)
        # print('--> FORWARD PROPAGATION:', x.size(), x.size(0))



        in_size = x.size(0)



        # --> Pass inputs through first convolutional layer then apply activation function ReLU
        x1 = F.relu(self.conv1(x))
        # print(1, x1.size())

        # --> Pass output of convolutional layer x1 as the input for 2 x 2 maximum pooling
        x1 = self.mp(x1)  # size=(N, 32, x.H/2, x.W/2)
        # print(2, x1.size())



        # --> Pass inputs through second convolutional layer then apply activation function ReLU
        x2 = F.relu(self.conv2(x1))
        # print(3, x2.size())

        # --> Pass output of convolutional layer x2 as the input for 2 x 2 maximum pooling
        x2 = self.mp(x2)  # size=(N, 64, x.H/4, x.H/4)
        # print(4, x2.size())





        # --> Pass inputs through third convolutional layer then apply activation function ReLU
        x3 = F.relu(self.conv3(x2))
        # print(5, x3.size())

        # --> Pass output of convolutional layer x3 as the input for 2 x 2 maximum pooling
        x3 = self.mp(x3)  # size=(N, 128, x.H/8, x.H/8)
        # print(6, x3.size())




        # --> Transform output and pass through final linear nn layer
        x4 = x3.view(in_size, -1)

        # print('--> SHAPE BEFORE LINEAR TRANSFORM:', x4.size())
        x4 = self.fc(x4)  # size=(N, n_class)


        y = F.log_softmax(x4, dim=0)  # size=(N, n_class)

        # print('--> x1:', x1.size())
        # print('--> x2:', x2.size())
        # print('--> x3:', x3.size())
        # print('--> x4:', x4.size())
        # print('-->  y:', y.size())



        return x1, x2, x3, x4, y


class LossFunction(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LossFunction, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        print('--> LOSS CALCULATION:', inputs.size(), targets.size())



        inputs = inputs.view(-1)
        targets = targets.view(-1)
        print(1, inputs.size(), targets.size())

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice



class Encoding(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super(Encoding, self).__init__()


        self.n_class = n_class
        self.pretrained_net = pretrained_net


        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        # print('--> ENCODING FORWARD PROPAGATION:', x.size())
        output = self.pretrained_net(x)

        # print('--> DECODING FORWARD PROPAGATION:', x.size())
        x3 = output[2]
        x2 = output[1]
        x1 = output[0]


        # --> 1. Deconvolve
        score = self.relu(self.deconv1(x3))
        # print(1, score.size())

        # --> 2. Deconvolve
        score = self.bn1(score + x2)
        score = self.relu(self.deconv2(score))
        # print(2, score.size())

        # --> 3. Deconvolve
        score = self.bn2(score + x1)
        score = self.bn3(self.relu(self.deconv3(score)))
        # print(3, score.size())

        # --> 4. Classify score
        score = self.classifier(score)
        # print(4, score.size())

        return score

