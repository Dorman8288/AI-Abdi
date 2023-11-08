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
from Datasets import CustomFunctionDataset
import math
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader
from Models import BasicModel
import time

def Train(model, dataLoader, xrange, yrange, lossFunction, learningRate):
    fig, axes = plt.subplots(1, 1)
    axes: Axes = axes
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
    text = axes.text(0.03, 0.97, "", transform=axes.transAxes, va="top")
    DataPoints = np.linspace(xrange[0], xrange[1], (xrange[1] - xrange[0]) * 10)
    data_x = [x for x, y in dataLoader]
    correct = [y for x, y in dataLoader]
    correctFunction = axes.scatter(data_x, correct, color="blue")
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
TrainSize = 7
yrange = (2, 7)
xrange = (3, 16)
loss = nn.MSELoss()
learningRate = 0.01



dataset = CustomFunctionDataset([3.6, 4.7, 5.8, 6.7, 7.7, 8.4, 9.1, 10.7, 11.1, 12.4, 12.6, 13.6, 14.5, 14.7, 15.5],
                                [5.3, 6.6, 5.6, 4, 6.0, 8.2, 6.1, 6.6, 5.9, 6.4, 5.7, 6.2, 5.6, 3.7, 3])
dataLoader = DataLoader(dataset, batch_size=1)
Model = BasicModel([1, 50, 200, 200, 200, 100, 1], nn.ReLU(), None).to(device)
Train(Model, dataLoader, xrange, yrange, loss, learningRate)





