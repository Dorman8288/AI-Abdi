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