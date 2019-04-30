#Alexander Pavlak
#1001572620
#CSE 4334-001
#Homework 3: Logistic Regression

import numpy as np
from math import log
import matplotlib.pyplot as plt

#x = 1x3 np array representing a 2d data point and its label
#w = 1x3 np array representing the weights
def outputFunction(x,w):
    exponent = w[2] + (w[0]*x[0]) + (w[1]*x[1])
    result = 1 / (1 + np.exp(-exponent))
    return result


#Function to comput the Gradient
#x = sample vector
#label = the class label of the data point
#w = 1x3 np array representing the weights
def gradient(x,y,output):
    return (output - y) * x

def crossEntropy(output,label):
    result = (-label*log(output)) - ((1-label)*log(1-output))
    return result


def batch(trainingSet,trainingLabels,testingSet,testingLabels,rate):
    weights = np.random.uniform(-0.01, 0.01, 3)

    iterations = 0
    gradientNorm = 1
    entropys = []
    norms = []
    
    while(iterations < 100000 and gradientNorm > .001):
        error = np.zeros(3)
        entropy = 0
        for x,y in zip(trainingSet,trainingLabels):
            output =  outputFunction(x,weights)
            error += gradient(x,y,output)
            entropy += crossEntropy(output,y)
        #update weights
        error /= 1000
        entropys.append(entropy)
        weights += (-rate * error)
        print("Iteration: " + str(iterations))
        print(weights)
        #update loop variables
        iterations += 1
        gradientNorm = np.linalg.norm(error,ord=1)
        norms.append(gradientNorm)
        print("GradientNorm: "+ str(gradientNorm))

    print(error)
    slope = weights[0]/weights[1]
    intercept = weights[2]/weights[1]
    plt.title("Entropy vs Iteration")
    plt.plot(entropys)
    plt.show()
    plt.title("Gradient Norm vs Iteration")
    plt.plot(norms)
    plt.show()
    plt.scatter(trainingSet[:,0],trainingSet[:,1])
    plt.plot(trainingSet[:,0], (-slope*trainingSet[:,0] - intercept))
    plt.show()

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
plt.scatter(trainingSet0[:,0],trainingSet0[:,1])
plt.scatter(trainingSet1[:,0],trainingSet1[:,1])
plt.show()
#add labels to the data

trainingLabels0 = np.zeros((500,1))
trainingLabels1 = np.ones((500,1))
testingLabels0 = np.zeros((500,1))
testingLabels1 = np.ones((500,1))

#combine sets
trainingSet = np.concatenate((trainingSet0,trainingSet1),axis=0)
testingSet = np.concatenate((testingSet0,testingSet1),axis=0)
trainingLabels = np.concatenate((trainingLabels0,trainingLabels1),axis=0)
testingLabels = np.concatenate((testingLabels0,testingLabels1),axis=0)

#add bias term to the sets
trainingSet = np.append(trainingSet,np.ones((1000,1)),axis=1)
testingSet = np.append(testingSet,np.ones((1000,1)),axis=1)

print(trainingSet.shape)
print(testingSet.shape)

batch(trainingSet,trainingLabels,testingSet,testingLabels,1)