import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
from Datasets import AirPlaneDatabase
import math
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader
from Models import BasicModel
import time

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            for i in range(len(pred)):
                correct += (float(pred[i][0]) >= 0.5) == float(y[i])
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

batchsize = 100
learningRate = 0.01
epochs = 10


Trainset = AirPlaneDatabase(True)
Testset = AirPlaneDatabase(False)
TrainLoader = DataLoader(Trainset, batchsize, True)
TestLoader = DataLoader(Testset, batchsize, True)

model = BasicModel([32, 50, 50, 1], nn.ReLU(), nn.Sigmoid()).to("cpu")

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(TrainLoader, model, loss_fn, optimizer)
    test_loop(TestLoader, model, loss_fn)
print("Done!")
