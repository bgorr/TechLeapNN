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









class NeuralNetClient:

    def __init__(self):

        self.device = 'gpu'

        self.batch_size = 1
        self.num_classes = 2
        self.num_epochs = 100
        self.threshold = torch.Tensor([.666]).to(self.device)



        torch.cuda.empty_cache()






        self.test = 0








