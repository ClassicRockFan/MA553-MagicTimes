#!/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend
import scipy.stats
import tensorflow as tf
from tensorflow import keras
from datetime import timedelta

import scipy.stats
import pickle

rideNames = [ 'Infinity Falls', 'Mako', 'Kraken', 'Manta', 'Abby\'s Flower Tower', 'Super Grover\'s Box Car Derby', 'Turtle Trek', 'Antarctica: Empire of the Penguin', 'Wild Arctic', 'Journey to Atlantis', 'Ice Breaker', 'Tigris', 'Cheetah Hunt', 'SheiKra', 'Montu', 'Falcon\'s Fury', 'Cobra\'s Curse', 'Kumba', 'Congo River Rapids', 'Scorpion', 'SandSerpent']

with open("seaworld_test.pkl",'rb') as f:
    test_data = pickle.load(f)  
model = tf.keras.models.load_model('seaworld')

inputData = test_data["x"]
outputData = test_data["y"]
allPredictions = model.predict(inputData)
#allPredictions = np.around(allPredictions/5, decimals=0)*5
# allResiduals = outputData - allPredictions

# rideResiduals = []
# for i in range(0, len(rideNames)):
#     rideResiduals.append([])
# for i in range(0, len(inputData)):
#     for j in range(0, len(rideNames)):
#         ride = allResiduals[i][j]
#         for k in range(0, len(ride)):
#             rideResiduals[j].append(ride[k])
# for i in range(0, len(rideNames)):
#     plt.hist(rideResiduals[i], bins=20)
#     mean = np.average(rideResiduals[i])
#     std = np.std(rideResiduals[i])
#     dist = scipy.stats.norm(mean, std)
#     p = dist.cdf(10) - dist.cdf(-10)
#     print(rideNames[i] + ": avg = {0:.2f}; std = {1:.2f}; p +- 10 minutes: {2:0.2f}".format(mean, std, p))
#     plt.title(rideNames[i])
#     plt.savefig(rideNames[i] + ".png")    
    #plt.show()

# example = model.predict(inputData)
# print(inputData[0])
# print(example[0])

timeIndex = 40
rides = [0, 1, 2, 3, 4, 5, 6, 7]
for timeIndex in range(0, 500):
    for rideIndex in rides:
        maxVal = max(np.max(inputData[timeIndex][rideIndex]), np.max(outputData[timeIndex][rideIndex]), np.max(allPredictions[timeIndex][rideIndex]))
        if(maxVal == 0):
            continue
        plt.ylim([0, maxVal + 5])
        plt.yticks(np.arange(0, maxVal+5, step=5))
        plt.plot(range(0, 60, 5), inputData[timeIndex][rideIndex])
        plt.plot(range(60, 120, 5), outputData[timeIndex][rideIndex])
        plt.plot(range(60, 120, 5), allPredictions[timeIndex][rideIndex])
        plt.legend(["Input Behaviour", "True Output", "Predicted Output"])
        plt.title(rideNames[rideIndex])
        print(rideNames[rideIndex])
        plt.savefig(rideNames[rideIndex] + "_predictions.png")    
        plt.show()
    