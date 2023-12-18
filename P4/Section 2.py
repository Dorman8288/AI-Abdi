import numpy as np
import numpy.random as rand
from sklearn import svm
import matplotlib.pylab as plt
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.widgets import Slider
import cv2
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        label = filename[0]
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.flatten())
            labels.append(label)
    return (images, labels)

testFolder = "./Data/USPS_images/images/test"
trainFolder = "./Data/USPS_images/images/train"

testData = load_images_from_folder(testFolder)
trainData = load_images_from_folder(trainFolder)

gamma = 0.5
C = 10

classifier = svm.SVR(kernel="rbf", gamma=gamma, C=C, cache_size=700)
classifier.fit(trainData[0], trainData[1])

print(classifier.score(testData[0], testData[1]))

correct = 0
print(testData[0])
y = classifier.predict(testData[0])
print(y)
for i in range(len(testData[0])):
    correct += testData[1][i] == y[i]
print(f"Accuracy: {correct/len(testData)}")