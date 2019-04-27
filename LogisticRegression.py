#Alexander Pavlak
#1001572620
#CSE 4334-001
#Homework 3: Logistic Regression

import numpy as np
import matplotlib.pyplot as plt

def outputFunction(x,w):
    exponent = w[0] + (w[1]*x[0]) + (w[2]*x[1])
    result = 1 / (1 + exp(-exponent))

def errorFunction():

def batch(trainingSet,testingSet):
    weights = np.random.uniform(low=-0.01,high=0.01,3)

    iterations = 0
    #gradient norm needs to be replaced by the l1 norm of the gradient
    while(iterations < 100000 or gradientNorm<.001):
       error = 0
       for i in trainingSet:
           output =  outputFunction(i,weights)
           error += .5 * (i[2] - output)**2
           gradient = np.array()

#Given parameters for generating 1st set of gausian numbers
mean1 = np.array([1,0])
dev1 = np.array([[1.0,0.75],[0.75,1.0]])

#Given parameters for generating 2nd set of gausian numbers
mean2 = np.array([0,1.5])
dev2 = ([[1.0,0.75],[0.75,1.0]])

#Generate random training and testing data
trainingSet0 = np.random.multivariate_normal(mean1,dev1,500)
trainingSet1 = np.random.multivariate_normal(mean2,dev2,500)
testingSet0 = np.random.multivariate_normal(mean1,dev1,500)
testingSet1 = np.random.multivariate_normal(mean2,dev2,500)

#add labels to the data

trainingSet0 = np.append(trainingSet0,np.zeros((500,1)),axis=1)
trainingSet1 = np.append(trainingSet1, np.ones((500,1)),axis=1)
testingSet0 = np.append(testingSet0,np.zeros((500,1)),axis=1)
testingSet1 = np.append(testingSet1, np.ones((500,1)),axis=1)

#combine sets
trainingSet = np.concatenate((trainingSet0,trainingSet1),axis=0)
testingSet = np.concatenate((testingSet0,testingSet1),axis=0)

print(trainingSet)
print(testingSet)