import pandas as pd
import numpy
from DecisionTree import *
import copy
import matplotlib.pyplot as plt
import time

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

def Categorization1(TrainData, TestData, features, categorizedFeatures):
    featuresDomains = {}
    for feature in features:
        if feature == "id" or feature == "Unnamed: 0":
            continue
        if feature in categorizedFeatures:
            TrainData[feature], bins = categorize(TrainData[feature], 0.25)
            TestData[feature] = series = pd.cut(TestData[feature], bins, duplicates="drop").astype(str)
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
        



def CleanDataV1(TrainData, TestData, features, categorizedFeatures):
    featureDomains = Categorization1(TrainData, TestData, features, categorizedFeatures)
    EncodeData(TestData, copy.deepcopy(featureDomains))
    EncodeData(TrainData, featureDomains)
    return featureDomains


def Train(TrainData, featuresDomains, depth, importance):
    return DesicionTree(featuresDomains, TrainData, featuresDomains["satisfaction"], depth, set(["satisfaction"]), importance)

def Test(TestData, tree):
    correct = 0
    for i in range(len(TestData)):
        entry = TestData.iloc[i]
        correct += tree.classify(entry) == TestData.iloc[i]["satisfaction"]
    return (correct / len(TestData)) * 100

DataPath = "Data/Airplane.csv"
DataFrame = pd.read_csv(DataPath)
TestData = DataFrame.iloc[:2000]
TrainData = DataFrame.iloc[2000:]
features = TestData.keys()
categorizedFeatures = set(["Age",
                           "Flight Distance",
                           "Departure Delay in Minutes",
                           "Arrival Delay in Minutes"])
featuresDomains = CleanDataV1(TrainData, TestData, features, categorizedFeatures)
depths = []
GiniAccuracies = []
GainAccuracies = []
n = len(featuresDomains)
for i in range(4, 7):
    giniTime = time.perf_counter()
    tree = Train(TrainData, featuresDomains, i, DecisionNode.calculateGini)
    giniTime = time.perf_counter() - giniTime
    GiniAccuracy = Test(TestData, tree)
    gainTime = time.perf_counter()
    tree = Train(TrainData, featuresDomains, i, DecisionNode.calculateEntropy)
    gainTime = time.perf_counter() - gainTime
    GainAccuracy = Test(TestData, tree)
    depths.append(i)
    GiniAccuracies.append(GiniAccuracy)
    GainAccuracies.append(GainAccuracy)
    print(f"Test Accuracy with Gain {i}: {GainAccuracy}%")
    print(f"Test Accuracy with Gini {i}: {GainAccuracy}%")
    print(f"Gain Train Time{i}: {gainTime}")
    print(f"Gini Train Time{i}: {giniTime}%")







