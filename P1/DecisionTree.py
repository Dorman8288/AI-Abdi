
import numpy
import pandas as pd
import math

class Node:
    def __init__(self) -> None:
        pass


class DecisionNode(Node):
    def __init__(self, data: pd.DataFrame, featureDomain: dict, labels, depth, usedFeatures: set, importance):
        self.Childs = {}
        self.importance = importance
        self.data = data
        self.domains = featureDomain
        self.labels = labels
        self.depth = depth
        self.used = usedFeatures
        self.feature = self.pickBestFeature(importance)
        self.used.add(self.feature)
        self.makeChilds()
        self.used.remove(self.feature)

    def pickBestFeature(self, importance):
        bestFeature = None
        MaximumGain = -math.inf 
        for feature in self.domains:
            if feature not in self.used:
                gain = self.calculateGain(feature, importance)
                if MaximumGain < gain:
                    MaximumGain = gain
                    bestFeature = feature
        return bestFeature

    def makeChilds(self):
        for item in self.domains[self.feature]:
            nextData = query(self.data, self.feature, item)
            if len(nextData) != 0:
                entropy = self.importance(self.labels, nextData)
            if len(nextData) == 0:
                self.Childs[item] = LeafNode(self.labels, self.data)
            elif entropy == 0 or self.depth == 0:
                self.Childs[item] = LeafNode(self.labels, nextData)
            else:
                self.Childs[item] = DecisionNode(nextData, self.domains, self.labels, self.depth - 1, self.used, self.importance)

    def Visualize(self, tabs):
        indent = "---" * tabs
        print(indent + self.feature)
        for value in self.Childs:
            print(indent + "-" + f"{value}" + ':')
            self.Childs[value].Visualize(tabs + 1)

    def calculateGini(labels, data: pd.DataFrame):
        Gini = 0
        for label in labels:
            prob = len(query(data, "satisfaction", label)) / len(data)
            if prob == 0:
                continue
            Gini += prob * (1 - prob)
        return Gini

    def calculateGain(self, feature, importance):
        totalChildImportance = 0
        domain = self.domains[feature]
        for item in domain:
            nextData = query(self.data, feature, item)
            if len(nextData) > 0:
                totalChildImportance += importance(self.labels, nextData)
        averageChildImportance = totalChildImportance / len(domain)
        parentImportance = importance(self.labels, self.data)
        return parentImportance - averageChildImportance

    def calculateEntropy(labels, data: pd.DataFrame):
        entorpy = 0
        for label in labels:
            prob = len(query(data, "satisfaction", label)) / len(data)
            if prob == 0:
                continue
            entorpy += prob * math.log2(prob)
        return -entorpy
    
    def classify(self, entry):
        value = entry[self.feature]
        return self.Childs[value].classify(entry)


class LeafNode(Node):

    def Visualize(self, tabs):
        indent = "---" * tabs
        print(indent, self.probs)

    def GetProbabilities(self, Data: pd.DataFrame, lables: set):
        probabilities = []
        TotalCount = len(Data)
        for label in lables:
            prob = len(query(Data, "satisfaction", label)) / TotalCount
            probabilities.append((label, prob))
        return probabilities
            

    def __init__(self, labels: set, data: pd.DataFrame):
        self.probs = self.GetProbabilities(data, labels)
        self.ranges = self.makeRanges(self.probs)
        self.labels = labels


    def makeRanges(self, probs):
        ranges = []
        total = 0
        for label, prob in probs:
            total += prob
            ranges.append((label, total))
        return ranges
    
    def classify(self, entry):
        pick = numpy.random.uniform(0.0, 1)
        for i in range(len(self.labels)):
            label, border = self.ranges[i]
            if pick <= border:
                return label


def query(data: pd.DataFrame, column, value):
    return data.loc[data[column] == value]

class DesicionTree():
    def __init__(self, featureDomains, Data, labels, depth, ignore, importance):
        self.Data = Data
        self.featureDomains = featureDomains
        self.depth = depth
        self.root = DecisionNode(self.Data, featureDomains, labels, depth, ignore, importance)
    
    def classify(self, entry):
        return self.root.classify(entry)
        


        
        