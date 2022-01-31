from processing.api import DataProcessingClient
from neuralnet.api import NeuralNetClient


import numpy as np
import pprint
from scipy import interpolate
import torch





def processing():

    # --> 2. Process test data
    data_dir = '/home/ec2-user/repos/TechLeapNN/data/viirs/test_dataset'
    save_path = '/home/ec2-user/repos/TechLeapNN/output/test_dataset_7b_200.p'
    processing_client = DataProcessingClient(data_dir=data_dir, save_path=save_path)
    processing_client.build()
    processing_client.run()

    # --> 1. Process training data
    data_dir = '/home/ec2-user/repos/TechLeapNN/data/viirs/training_dataset'
    save_path = '/home/ec2-user/repos/TechLeapNN/output/training_dataset_7b_200.p'
    processing_client = DataProcessingClient(data_dir=data_dir, save_path=save_path)
    processing_client.build()
    processing_client.run()







if __name__ == "__main__":
    processing()




