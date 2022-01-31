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

from neuralnet.models.VanillaCNN import VanillaCNN, Encoding, LossFunction







class NeuralNetClient:

    def __init__(self, training_dataset, test_dataset, result_file):
        random.seed(1319)
        matplotlib.use('Agg')

        # --> Model Parameters
        self.device = 'cuda'
        self.batch_size = 5
        self.num_classes = 2
        self.threshold = torch.Tensor([.666]).to(self.device)

        # --> Data
        self.result_file = result_file

        self.training_dataset = pickle.load(open(training_dataset, 'rb'))
        self.test_dataset = pickle.load(open(test_dataset, 'rb'))

        self.training_tensor = torch.utils.data.DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_tensor = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # --> Model
        self.cnn = VanillaCNN().to(self.device)
        self.model = Encoding(pretrained_net=self.cnn, n_class=self.num_classes).to(self.device)
        self.loss_func = LossFunction()

        # --> Stochastic gradient descent
        self.optimizer = optim.SGD(self.model.parameters(), lr=3e-4, momentum=0.5)

        self.weights = self.build_weights()


        torch.cuda.empty_cache()

    def build_weights(self):
        weights = torch.tensor([1, 9], dtype=torch.float32)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        return weights

    @property
    def test_tensor_data(self):
        return [d for d, _ in self.test_tensor]




    """
     _______           _        
    |__   __|         (_)       
       | | _ __  __ _  _  _ __  
       | || '__|/ _` || || '_ \ 
       | || |  | (_| || || | | |
       |_||_|   \__,_||_||_| |_|                      
    """

    def train(self, epochs=20, save=True, plot=True):

        # --> 1. Train desired number of epochs
        for epoch in range(epochs):
            self._train(epoch)

        # --> 2. Get results of trained model
        results = self.test(doSave=True, threshold=self.threshold, epoch=epochs-1)

        # --> 3. Save final epoch results
        if results is not None:
            if save is True:
                self.save(results)
            if plot is True:
                self.plot(results)

    def _train(self, epoch):

        # --> 1. Train the model
        self.model.train()

        # --> 2. Iterate over training data tensor
        for batch_idx, (data, target) in enumerate(self.training_tensor):
            data, target = Variable(data).float().to(self.device), Variable(target).float().to(self.device)
            data = torch.cuda.FloatTensor(data)


            self.optimizer.zero_grad()

            output = self.model(data)
            # print('--------------> GROUND TRUTH:', target.size())
            # print('----------> MODEL PREDICTION:', output.size())

            output2 = torch.sigmoid(output)
            # print('--> MODEL PREDICTION SIGMOID:', output2.size())

            loss = self.loss_func(output2, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.training_tensor.dataset),
                           100. * batch_idx / len(self.training_tensor), loss.item()))





    def save(self, results):
        f = open(self.result_file, 'wb')
        pickle.dump(results, f)
        f.close()

    def plot(self, results):
        for i in range(len(results[0])):
            X = self.test_tensor_data[i].cpu()
            Y = results[1][i].cpu()
            Yest = results[2][i].cpu()
            Yhat = results[3][i].cpu()
            Z = results[0][i].cpu()
            for j in range(len(X)):
                blue = X[j, 0, :, :].detach().numpy()
                green = X[j, 1, :, :].detach().numpy()
                red = X[j, 2, :, :].detach().numpy()
                ir = X[j, 3, :, :].detach().numpy()
                swir = X[j, 4, :, :].detach().numpy()
                clouds = X[j, 5, :, :].detach().numpy()
                temp = X[j, 6, :, :].detach().numpy()
                img = np.array([red, blue, green])
                img = np.moveaxis(img, 0, -1)
                for n in range(3):
                    mx = np.max(img[:, :, n])
                    mn = np.min(img[:, :, n])
                    img[:, :, n] = (img[:, :, n] - mn) / (mx - mn)
                y = Y[j, :, :].detach().numpy()
                z = Z[j, :, :].detach().numpy()
                yhat = Yhat[j, :, :].detach().numpy()
                yest = Yest[j, :, :].detach().numpy()
                plt.subplot(2, 5, 1)
                plt.imshow(img)
                plt.subplot(2, 5, 2)
                plt.imshow(red)
                plt.subplot(2, 5, 3)
                plt.imshow(ir)
                plt.subplot(2, 5, 4)
                plt.imshow(swir)
                plt.subplot(2, 5, 5)
                plt.imshow(clouds)
                plt.subplot(2, 5, 6)
                plt.imshow(temp)
                plt.subplot(2, 5, 7)
                plt.imshow(z)
                plt.subplot(2, 5, 8)
                plt.imshow(y)
                plt.subplot(2, 5, 9)
                plt.imshow(yhat)
                plt.subplot(2, 5, 10)
                plt.imshow(yest)
                plt.savefig('./allplots/fig{}{}.png'.format(i, j))
                plt.close('all')


    """
     _______          _   
    |__   __|        | |  
       | |  ___  ___ | |_ 
       | | / _ \/ __|| __|
       | ||  __/\__ \| |_ 
       |_| \___||___/ \__|
    """

    def test(self, doSave, threshold, epoch):
        print('\n--> TESTING MODEL: initializing')
        self.model.eval()

        acc = []

        iou_mn = []
        iou_tp = []
        iou_tn = []

        wiou_tp = []
        wiou_tn = []

        all_targets = []
        all_out = []
        all_output = []
        all_pred = []



        # --> Iterate over test dataset
        counter = 0
        for data, target in self.test_tensor:
            counter += 1
            print('--> Test image:', counter, data.size())
            print('--> Test label:', counter, target.size())

            # --> Image patch pixel data (p, c, 161, 105) -> [64, 7, 161, 105]
            data = Variable(data).float().to(self.device)

            # --> Label patch pixel data (p, c, 161, 105) -> [64, 2, 161, 105]
            target = Variable(target).float().to(self.device)

            out = self.model(data)
            output = torch.sigmoid(out)
            pred = (output[:, 1, :, :] > threshold).float() * 1
            acc.append(self.calc_acc(p=pred.long(), t=target[:, 1, :, :].long()))
            iou_out = self.calc_wiou(p=pred, t=target[:, 1, :, :])
            wiou_tp.append(iou_out[0])
            wiou_tn.append(iou_out[1])
            iou_tp.append(iou_out[2])
            iou_tn.append(iou_out[3])
            iou_mn.append(iou_out[4])
            if doSave:
                all_targets.append(target[:, 1, :, :])
                all_out.append(out[:, 1, :, :])
                all_output.append(output[:, 1, :, :])
                all_pred.append(pred)
        aa = np.array(acc).mean()
        bb = self.listMean(iou_mn, 1)
        cc = self.listMean(iou_tp, 1)
        dd = self.listMean(wiou_tp, 2)
        ee = self.listMean(iou_tn, 1)
        ff = self.listMean(wiou_tn, 2)

        print('\n--> Testing model:', epoch, 'epochs <--')
        print('-------> Accuracy:', str(round(100 * aa, 2)) + '%')
        print('------------> IoU:', str(round(100 * bb, 2)) + '%')
        print('--> True Positive:', str(round(100 * dd, 2)) + '%')
        print('--> True Negative:', str(round(100 * ff, 2)) + '%')



        print('\nTest Epoch: {}\t Accuracy: {}%, IoU: {}%, TP IoU: {}({})%, TN IoU = {}({})%\n'.format(
            epoch, round(100 * aa, 2), round(100 * bb, 2), round(100 * cc, 2), round(100 * dd, 2),
            round(100 * ee, 2), round(100 * ff, 2)))
        return all_targets, all_out, all_output, all_pred

    def calc_acc(self, p, t):
        p = p.cpu()
        t = t.cpu()
        correct_pixels = (p == t).sum().to(dtype=torch.float)
        total_pixels = (t == t).sum().to(dtype=torch.float)
        return correct_pixels / total_pixels

    def calc_iou(self, p, t):
        p = p.cpu()
        t = t.cpu()
        current = confusion_matrix(t.numpy().flatten(), p.numpy().flatten(), labels=[0, 1])
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)  # rows
        predicted_set = current.sum(axis=0)  # columns
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)
        tp = IoU[1]
        tn = IoU[0]
        if math.isnan(tp):
            tp = 0
        if math.isnan(tn):
            tn = 0
        iou = (tp + tn) / 2
        return tp, tn, iou

    def calc_wiou(self, p, t):
        N = len(t)
        p = p.cpu()
        t = t.cpu()
        w_TP = np.zeros(N)
        w_TN = np.zeros(N)
        IoU = np.zeros(N)
        IoU_TP = np.zeros(N)
        IoU_TN = np.zeros(N)
        for j in range(N):
            pred, tar = p[j], t[j]
            w_TP[j] = sum(sum(tar))
            w_TN[j] = sum(sum(1 - tar))
            IoU_TP[j], IoU_TN[j], IoU[j] = self.calc_iou(p=pred, t=tar)
        wIoU_TP = sum(w_TP * IoU_TP) / sum(w_TP) ###
        if math.isnan(wIoU_TP):
            wIoU_TP = 0
        wIoU_TN = sum(w_TN * IoU_TN) / sum(w_TN)
        if math.isnan(wIoU_TN):
            wIoU_TN = 0
        return wIoU_TP, wIoU_TN, IoU_TP, IoU_TN, IoU

    def listMean(self, my_list, mean_type):
        if mean_type == 1:
            avg1 = [float(sum(col)) / len(col) for col in zip(*my_list)]
            return sum(avg1) / len(avg1)
        if mean_type == 2:
            return sum(my_list) / len(my_list)










