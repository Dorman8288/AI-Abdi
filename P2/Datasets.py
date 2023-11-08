import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
import math
import copy

class FunctionDataset(Dataset):
    def __init__(self, Function, Count, Range):
        self.Function = Function
        self.Count = Count
        self.Data = []
        # if withNoise:
        #     noise = np.random.normal(0, 1, Count)
        # else:
        #     noise = [0 for i in range(Count)]
        for i in range(Count):
            rand = random.uniform(Range[0], Range[1])
            self.Data.append((rand, Function(rand)))

    def __len__(self):
        return self.Count

    def __getitem__(self, idx):
        x, y = self.Data[idx]
        return (torch.tensor([x], dtype=torch.float64), torch.tensor([y], dtype=torch.float64))
    
    
class RandomFunctionDataset(Dataset):
    def __init__(self, Count, Range):
        self.datapoints = np.linspace(Range[0], Range[1], Count)
        self.Count = Count
        self.Data = []
        for i in range(Count):
            rand = random.uniform(Range[0], Range[1])
            self.Data.append((self.datapoints[i], rand))

    def __len__(self):
        return self.Count

    def __getitem__(self, idx):
        x, y = self.Data[idx]
        return (torch.tensor([x], dtype=torch.float64), torch.tensor([y], dtype=torch.float64))
    
class CustomFunctionDataset(Dataset):
    def __init__(self, X, Y):
        self.Count = len(X)
        self.Data = []
        for i in range(self.Count):
            self.Data.append((X[i], Y[i]))

    def __len__(self):
        return self.Count

    def __getitem__(self, idx):
        x, y = self.Data[idx]
        return (torch.tensor([x], dtype=torch.float64), torch.tensor([y], dtype=torch.float64))
    
class AirPlaneDatabase(Dataset):
    def __init__(self, train) -> None:
        super().__init__()
        self.dataset = pd.read_csv("Data/Airplane.csv")
        if not train:
            self.dataset = self.dataset.iloc[:2000]
        else:
            self.dataset = self.dataset.iloc[2000:]
        self.dataset = self.dataset.drop("id", axis="columns")
        self.dataset = self.dataset.drop("Unnamed: 0", axis="columns")
        self.y = self.dataset["satisfaction"]
        self.dataset.drop("satisfaction", axis="columns")
        features = self.dataset.keys()
        continousFeatures = set(["Age",
                           "Flight Distance",
                           "Departure Delay in Minutes",
                           "Arrival Delay in Minutes"])
        categorizedFeatures = set(["Gender",
                           "Customer Type",
                           "Type of Travel",
                           "Class"])
        self.dataset = CleanDataV1(self.dataset, features, categorizedFeatures, continousFeatures)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = torch.tensor(list(self.dataset.iloc[idx]), dtype=float)
        y = torch.tensor([1] if self.y.iloc[idx] == "satisfied" else [0], dtype=torch.float64)
        return x, y




def categorize(series: pd.Series, precentInGroup):
    CountInGroup = int(len(series) * precentInGroup)
    count = int(1 / precentInGroup)
    series = series.sort_values()
    bins = [-math.inf]
    border = 0
    for i in range(int(count)):
        bins.append(series.iloc[border])
        border += CountInGroup
    bins.append(math.inf)
    series = pd.cut(series, bins, duplicates="drop")
    out = series.astype(str).sort_index()
    return (out, bins)

def Categorization1(TrainData, features, categorizedFeatures):
    featuresDomains = {}
    for feature in features:
        if feature == "id" or feature == "Unnamed: 0":
            continue
        if feature in categorizedFeatures:
            TrainData[feature], bins = categorize(TrainData[feature], 0.25)
        series = TrainData[feature]
        domain = set()
        for item in series:
            if item not in domain:
                domain.add(item)
        featuresDomains[feature] = domain
    plt.show()
    return featuresDomains

def EncodeData(data: pd.DataFrame, featureDomains):
    for feature in featureDomains:
        domain = featureDomains[feature]
        featureDomains[feature] = list(range(len(domain)))
        ReplaceList = {}
        for encoding, item in enumerate(domain):
            ReplaceList[item] = encoding
        data[feature] = data[feature].replace(ReplaceList)

def CleanDataV1(TrainData, features, categorizedFeatures, continousFeatures):
    featureDomains = Categorization1(TrainData, features, continousFeatures)
    EncodeData(TrainData, featureDomains)
    for feature in categorizedFeatures:
        onehot = pd.get_dummies(TrainData[feature], dtype=float)
        TrainData.drop(feature, axis="columns")
        TrainData = pd.concat([TrainData, onehot], axis=1)
    return TrainData