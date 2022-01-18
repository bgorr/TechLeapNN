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

warnings.filterwarnings("error")


# encoding
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(33280, 2)

    def forward(self, x):
        in_size = x.size(0)
        x1 = F.relu(self.conv1(x))
        x1 = self.mp(x1)  # size=(N, 32, x.H/2, x.W/2)
        x2 = F.relu(self.conv2(x1))
        x2 = self.mp(x2)  # size=(N, 64, x.H/4, x.H/4)
        x3 = F.relu(self.conv3(x2))
        x3 = self.mp(x3)  # size=(N, 128, x.H/8, x.H/8)
        x4 = x3.view(in_size, -1)
        x4 = self.fc(x4)  # size=(N, n_class)
        y = F.log_softmax(x4, dim=0)  # size=(N, n_class)
        return x1, x2, x3, x4, y


# encoding/decoding
class FCN8s(nn.Module):
    # Po-Chih Huang
    def __init__(self, pretrained_net, n_class):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x3 = output[2]
        x2 = output[1]
        x1 = output[0]
        score = self.relu(self.deconv1(x3))
        score = self.bn1(score + x2)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x1)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.classifier(score)
        return score


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).float().to(DEVICE), Variable(target).float().to(DEVICE)
        data = torch.FloatTensor(data)
        optimizer.zero_grad()
        output = model(data)
        output2 = torch.sigmoid(output)
        loss = loss_fn(output2, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# IoU for binary case
def iou(p, t):
    current = confusion_matrix(t.numpy().flatten(), p.numpy().flatten(), labels=[0, 1])
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)  # rows
    predicted_set = current.sum(axis=0)  # columns
    union = ground_truth_set + predicted_set - intersection
    try:
        IoU = intersection / union.astype(np.float32)
    except RuntimeWarning:
        print("invalid value encountered in true_divide")
        IoU = [0, 1]
    return np.mean(IoU), current, IoU[0], IoU[1]


# accuracy for binary case
def pixel_acc(p, t):
    p = p.cpu()
    t = t.cpu()
    correct_pixels = (p == t).sum().to(dtype=torch.float)
    total_pixels = (t == t).sum().to(dtype=torch.float)
    return correct_pixels / total_pixels


# test
def test(doSave, threshold):
    model.eval()
    n_batches = 0
    total_acc = []
    iou_mn = []
    iou_tp = []
    iou_tn = []
    all_targets = []
    all_out = []
    all_output = []
    all_pred = []
    for data, target in test_loader:
        n_batches += 1
        data, target = \
            Variable(data).float().to(DEVICE), Variable(target).float().to(DEVICE)
        out = model(data).to(DEVICE)
        output = torch.sigmoid(out).to(DEVICE)
        pred = (output[:, 1, :, :] > threshold).float() * 1
        total_acc.append(pixel_acc(p=pred.long(), t=target[:, 1, :, :].long()))
        iou_output = iou(p=pred.long(), t=target[:, 1, :, :].long())
        iou_mn.append(iou_output[0])
        iou_tn.append(iou_output[2])
        iou_tp.append(iou_output[3])
        if doSave:
            all_targets.append(target[:, 1, :, :])
            all_out.append(out[:, 1, :, :])
            all_output.append(output[:, 1, :, :])
            all_pred.append(pred)
    print('\nTest Epoch: {}\t Mean Batch Accuracy: {}%, Mean Batch IoU: {}%, TP IoU: {}%, TN IoU = {}%\n'.format(
        epoch, round(100 * np.array(total_acc).mean(), 2), round(100 * np.array(iou_mn).mean(), 2),
        round(100 * np.array(iou_tp).mean(), 2), round(100 * np.array(iou_tn).mean(), 2)))
    return all_targets, all_out, all_output, all_pred


torch.cuda.empty_cache()
DEVICE = "cpu"
# network settings
batch_size = 1
n_class = 2
n_epochs = 1

# set threshold
thres = torch.Tensor([.666]).to(DEVICE)  # try: 0, -.2, -.1, .1, .2, .3, .4
flnm = "666"

test_filename = "./output/test_dataset_ready.p"
train_filename = "./output/test_dataset_ready.p"
test_f = open(test_filename, "rb")
train_f = open(train_filename, "rb")
test_dataset_l = pickle.load(test_f)
train_dataset_l = pickle.load(train_f)
test_f.close()
train_f.close()

# data loader
random.seed(1319)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset_l, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset_l, batch_size=batch_size, shuffle=False)

# initialize model
cnn_model = CNN().to(DEVICE)
model = FCN8s(pretrained_net=cnn_model, n_class=n_class).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.5)
loss_fn = nn.BCELoss()

# run for NT Data Set
for epoch in range(n_epochs):
    train(epoch)
    results = test(doSave=((epoch + 1) == n_epochs), threshold=thres)

test_dat = []
for dat, _ in test_loader:
    test_dat.append(dat)

matplotlib.use('Agg')

# plotting
for i in range(len(results[0])):
    X = test_dat[i]
    Y = results[1][i]
    Yest = results[2][i]
    Yhat = results[3][i]
    Z = results[0][i]
    for j in range(len(X)):
        blue = X[j, 0, :, :].detach().numpy()
        green = X[j, 1, :, :].detach().numpy()
        red = X[j, 2, :, :].detach().numpy()
        ir = X[j, 3, :, :].detach().numpy()
        swir = X[j, 4, :, :].detach().numpy()
        # clouds = X[j, 5, :, :].detach().numpy()
        # temp = X[j, 6, :, :].detach().numpy()
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
        # plt.ion()
        plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.subplot(2, 4, 2)
        plt.imshow(red)
        plt.subplot(2, 4, 3)
        plt.imshow(ir)
        plt.subplot(2, 4, 4)
        plt.imshow(swir)
        plt.subplot(2, 4, 5)
        plt.imshow(z)
        plt.subplot(2, 4, 6)
        plt.imshow(y)
        plt.subplot(2, 4, 7)
        plt.imshow(yhat)
        plt.subplot(2, 4, 8)
        plt.imshow(yest)
        plt.savefig('./allplots/fig{}{}.png'.format(i, j))
        plt.close('all')

results_filename = './output/results.p'
results_f = open(results_filename, 'wb')
pickle.dump(results, results_f)
results_f.close()
