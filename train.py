from processing.api import DataProcessingClient
from neuralnet.api import NeuralNetClient


import numpy as np
import pprint
from scipy import interpolate
import torch





def train():

    print('--> TESTING NEURAL NET')

    training_dataset = './output/training_dataset_7b.p'
    test_dataset = './output/test_dataset_7b.p'
    result_file = './output/results.p'


    client = NeuralNetClient(training_dataset, test_dataset, result_file)

    client.train(save=True, plot=False)


if __name__ == "__main__":
    train()




