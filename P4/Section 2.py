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
        label = int(filename[0])
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(list(img.flatten()))
            labels.append(label)
    return (images, labels)

testFolder = "./Data/USPS_images/images/test"
trainFolder = "./Data/USPS_images/images/train"

testData = load_images_from_folder(testFolder)
trainData = load_images_from_folder(trainFolder)

classifier = svm.SVC(kernel="rbf", cache_size=700)
classifier.fit(trainData[0], trainData[1])

correct = 0
y = classifier.predict(testData[0])
for i in range(len(testData[0])):
    correct += testData[1][i] == y[i]
print(f"Accuracy: {correct/len(testData[0])}")

