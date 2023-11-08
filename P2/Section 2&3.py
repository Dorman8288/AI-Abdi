import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
from Datasets import FunctionDataset
from Datasets import RandomFunctionDataset
import math
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader
from Models import BasicModel
import time

def Train(model, dataLoader, function, xrange, yrange, lossFunction, learningRate):
    fig, axes = plt.subplots(1, 1)
    axes: Axes = axes
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
    text = axes.text(0.03, 0.97, "", transform=axes.transAxes, va="top")
    DataPoints = np.linspace(xrange[0], xrange[1], (xrange[1] - xrange[0]) * 10)
    correct = [function(x) for x in DataPoints]
    correctFunction, = axes.plot(DataPoints, correct, color="blue")
    learnedFunction, = axes.plot([], [], color="red")
    optimizer = torch.optim.SGD(model.parameters(), learningRate)
    lastupdate = time.time()
    plt.show(block=False)
    epoch = 0
    model.train()
    while True:
        for batch, (x, y) in enumerate(dataLoader):
            if(time.time() - lastupdate > 0.01):
                model.train()
                pred = model(x)
                loss = lossFunction(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.state_dict()
                optimizer.step()
                model.eval()
                predicted = [float(model(torch.tensor([x]))[0]) for x in DataPoints]
                learnedFunction.set_data(DataPoints, predicted)
                text.set_text(f"epoch: {epoch}\nbatch: {batch}\nloss: {loss}")
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(1e-3)
                lastupdate = time.time()
        epoch += 1

def FunctionWithNoise(function, Amount, intensity, range):
    noise = np.random.normal(Amount, intensity, (range[1] - range[0]) * 1000)
    def Function(x):
        return function(x) + noise[int(x * 1000)]
    return Function
        


device = "cpu"
TrainSize = 10000
yrange = (-10, 10)
xrange = (-7, 7)
function = FunctionWithNoise(math.sin, 0, 0.5, xrange)
loss = nn.MSELoss()
learningRate = 0.1


dataset = FunctionDataset(function, TrainSize, xrange)
dataLoader = DataLoader(dataset, batch_size=100)
Model = BasicModel([1, 10, 20, 10, 1], nn.ReLU(), nn.Tanh()).to(device)
Train(Model, dataLoader, function, xrange, yrange, loss, learningRate)





