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

rideNames = ['The Amazing Adventures of Spider-Man®', 'The Cat in The Hat™', 'Doctor Doom\'s Fearfall®', 'Dudley Do-Right\'s Ripsaw Falls®', 'Flight of the Hippogriff™', 'Harry Potter and the Forbidden Journey™', 'Jurassic Park River Adventure™', 'Popeye & Bluto\'s Bilge-Rat Barges®', 'Storm Force Accelatron®', 'The Incredible Hulk Coaster®', 'Hogwarts™ Express - Hogsmeade™ Station', 'Skull Island: Reign of Kong™', 'Hagrid\'s Magical Creatures Motorbike Adventure™', 'Despicable Me Minion Mayhem™', 'E.T. Adventure™', 'Hollywood Rip Ride Rockit™', 'MEN IN BLACK™ Alien Attack!™', 'Revenge of the Mummy™', 'Shrek 4-D', 'The Simpsons Ride™', 'TRANSFORMERS™: The Ride-3D', 'Harry Potter and the Escape from Gringotts™', 'Hogwarts™ Express - King\'s Cross Station']

with open("universal_test.pkl",'rb') as f:
    test_data = pickle.load(f)  
model = tf.keras.models.load_model('universal')

inputData = test_data["x"]
outputData = test_data["y"]
allPredictions = model.predict(inputData)
# #allPredictions = np.around(allPredictions/5, decimals=0)*5
# allResiduals = outputData - allPredictions

# rideResiduals = []
# for i in range(0, len(rideNames)):
    # rideResiduals.append([])
# for i in range(0, len(inputData)):
    # for j in range(0, len(rideNames)):
        # ride = allResiduals[i][j]
        # for k in range(0, len(ride)):
            # rideResiduals[j].append(ride[k])
# for i in range(0, len(rideNames)):
    # plt.hist(rideResiduals[i], bins=20)
    # mean = np.average(rideResiduals[i])
    # std = np.std(rideResiduals[i])
    # dist = scipy.stats.norm(mean, std)
    # p = dist.cdf(10) - dist.cdf(-10)
    # print(rideNames[i] + ": avg = {0:.2f}; std = {1:.2f}; p +- 10 minutes: {2:0.2f}".format(mean, std, p))
    # plt.title(rideNames[i])
    # plt.savefig(rideNames[i] + ".png")    
    # #plt.show()

# example = model.predict(inputData)
# print(inputData[0])
# print(example[0])

timeIndex = 35
rides = [0, 5, 6, 9, 12, 15, 16, 17, 20, 21]

for rideIndex in rides:
    maxVal = max(np.max(inputData[timeIndex][rideIndex]), np.max(outputData[timeIndex][rideIndex]), np.max(allPredictions[timeIndex][rideIndex]))
    print(maxVal)
    plt.ylim([0, maxVal + 5])
    plt.yticks(np.arange(0, maxVal+5, step=5))
    plt.plot(range(0, 60, 5), inputData[timeIndex][rideIndex])
    plt.plot(range(60, 120, 5), outputData[timeIndex][rideIndex])
    plt.plot(range(60, 120, 5), allPredictions[timeIndex][rideIndex])
    plt.legend(["Input Behaviour", "True Output", "Predicted Output"])
    plt.title(rideNames[rideIndex])
    plt.savefig(rideNames[rideIndex] + "_predictions.png")    
    plt.show()
