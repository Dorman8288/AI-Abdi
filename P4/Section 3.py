import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
from sklearn import svm
import torch.utils.data.dataloader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Grayscale()])

batch_size = 2
epoch = 10

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train = list(trainset)
test = list(testset)[:800]
print(len(train))
print("Data Proccessing Done")
trainData = ([node[0].flatten().numpy() for node in train], [node[1] for node in train])
testData = ([node[0].flatten().numpy() for node in test], [node[1] for node in test])
print("begining fiting")
classifier = svm.SVC(kernel="rbf", cache_size=1000)
classifier.fit(trainData[0], trainData[1])

correct = 0
y = classifier.predict(testData[0])
for i in range(len(testData[0])):
    correct += testData[1][i] == y[i]
print(f"Test Accuracy: {correct/len(testData[0])}")

correct = 0
y = classifier.predict(trainData[0])
for i in range(len(trainData[0])):
    correct += trainData[1][i] == y[i]
print(f"train Accuracy: {correct/len(trainData[0])}")