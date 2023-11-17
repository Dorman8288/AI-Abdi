import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
import math
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader
from Models import DenoisationModel
from Datasets import NoisyCifarDataSet
import time
import torchvision
from torchvision import transforms

def imshow(img, axes, fig):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  axes.set_data(np.transpose(npimg, (1, 2, 0)))
  fig.canvas.draw_idle()
  fig.canvas.start_event_loop(1e-3)

def Test(dataloader, model, loss_fn):
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
    return 100*correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def Train(model, dataLoader, validationLoader, visualizeBatch, learningRate, lossFunction):
    optimizer = torch.optim.Adam(model.parameters(), learningRate)
    epoch = 0
    model.train()
    subject, temp = validationLoader.dataset[visualizeBatch]
    subject = subject.reshape((1, 3, 32, 32))
    fig = plt.figure()
    axes = plt.imshow(torch.zeros((32, 32, 3)))
    #text = axes.text(0.03, 0.97, "", transform=axes.transAxes, va="top")
    fig.axes.append(axes)
    plt.show(block=False)
    lastupdate = time.time()
    while True:
        for batch, (image, noise) in enumerate(testloader):
            if(time.time() - lastupdate > 0.01):
                model.train()
                pred = model(image)
                loss = lossFunction(pred, noise)
                optimizer.zero_grad()
                optimizer.state_dict()
                loss.backward()
                optimizer.step()
                model.eval()
                #text.set_text(f"epoch: {epoch}\nbatch: {batch}\nloss: {loss}")
                noise = torch.Tensor.numpy(model(subject).detach())
                imshow(torchvision.utils.make_grid(subject - noise), axes, fig)
                lastupdate = time.time()
        #print(f"Validation Accuracy: {Test(validationLoader, model, lossFunction)}")
        epoch += 1




batchsize = 2
ImageSize = (32, 32)
learningRate = 0.01

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = NoisyCifarDataSet(transform, 0.5, 0.1, True)
testset = NoisyCifarDataSet(transform, 0.5, 0.1, False)

testloader = DataLoader(testset, batchsize, True)
trainloader = DataLoader(trainset, batchsize, True)

model = DenoisationModel().to("cpu")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

Train(model, trainloader, testloader, np.random.randint(0, len(testset)), learningRate, loss_fn)

# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(trainloader, model, loss_fn, optimizer)
#     test_loop(testloader, model, loss_fn)
# print("Done!")



# fig = plt.figure()
# axes = plt.imshow(torch.zeros((32, 32, 3)))
# fig.axes.append(axes)
# plt.show(block=False)
# lastupdate = time.time()
# for i in range(200):
#     for image, label in testloader:
#         if(time.time() - lastupdate > 0.1):
#             imshow(torchvision.utils.make_grid(image))
#             lastupdate = time.time()

