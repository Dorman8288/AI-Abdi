import numpy
from torch import nn
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
import math
from collections import OrderedDict
from torch.utils.data import DataLoader

class BasicModel(nn.Module):
    def __init__(self, widths, activation, end):
        super().__init__()
        temp = OrderedDict()
        self.flatten = nn.Flatten()
        for i in range(1, len(widths)):
            temp[str(2*i - 1)] = nn.Linear(widths[i - 1], widths[i], dtype=torch.float64)
            if i == len(widths) - 1:
                if end != None:
                    temp[str(2*i)] = end
                break
            temp[str(2*i)] = activation
        self.layers = nn.Sequential(temp)

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
class DenoisationModel(nn.Module):

    def __init__(self):
        super(ImageClassificationModel, self).__init__()

        self.flatten = nn.Flatten()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(160000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 101),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x 
    

class ImageClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.faltten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    

    def forward(self, x):
        x = self.layers(x)
        x = self.faltten(x)
        logits = self.classifier(x)
        return logits
    

class DenoisationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3, dtype=torch.float64),
            nn.BatchNorm2d(32, dtype=torch.float64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, padding=3, dtype=torch.float64),
            nn.BatchNorm2d(64, dtype=torch.float64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 7, padding=3, dtype=torch.float64),
            nn.BatchNorm2d(32, dtype=torch.float64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, padding=3, dtype=torch.float64),
            nn.BatchNorm2d(16, dtype=torch.float64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.faltten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(2048, 3072, dtype=torch.float64),
            nn.Sigmoid(),
            nn.Unflatten(1, unflattened_size=(3, 32, 32))
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.faltten(x)
        logits = self.classifier(x)
        return logits
    
        
