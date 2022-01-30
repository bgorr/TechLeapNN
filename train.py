from processing.api import DataProcessingClient
from neuralnet.api import NeuralNetClient


import numpy as np
import pprint
from scipy import interpolate
import torch





def train():

    print('--> TESTING NEURAL NET')

    client = NeuralNetClient()

    client.train(save=True, plot=False)


if __name__ == "__main__":
    train()




