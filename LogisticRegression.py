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

#Cross entropy function to measure the error of the predictions
#Output = ouput value of the data given by the outputFunction defined above
#label = class label for the data point. 
def crossEntropy(output,label):
    result = (-label*log(output)) - ((1-label)*log(1-output))
    return result

#prediction function
#w = trained weights
#testingSet = set of testing data to be classified
#testingLabels = set of labels for the testing data
def prediction(w,testingSet,testingLabels):
    prediction = 0
    correct = 0
    for i, j in zip(testingSet,testingLabels):
        output = outputFunction(i,w)
        if(output >=.5):
            prediction = 1
        else:
            prediction = 0
        if(prediction == j): correct += 1
    accuracy = correct / len(testingSet)
    return accuracy


def batch(trainingSet,trainingLabels,rate):
    weights = np.random.uniform(-0.01, 0.01, 3)

    iterations = 0
    gradientNorm = 1
    entropys = []
    norms = []
    print("Starting batch Training:")
    while(iterations < 100000 and gradientNorm > .001):
        if(iterations%100 == 0): print("Iteration: " + str(iterations))
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
        #update loop variables
        iterations += 1
        gradientNorm = np.linalg.norm(error,ord=1)
        norms.append(gradientNorm)
 

    fig, axs = plt.subplots(1,2)

    subTitle="Entropy vs Iteration"
    axs[0].set_title(subTitle)
    axs[0].plot(entropys)
   
    subTitle="Gradient Norm vs Iteration"
    axs[1].set_title(subTitle)
    axs[1].plot(norms)
    plt.suptitle('Training Rate = ' + str(rate))
    imageTitle = "batch_figures"+str(rate)+".png"
    plt.savefig(imageTitle)
    fig.clear()
    

    return weights

def online(trainingSet,trainingLabels,rate):
    weights = np.random.uniform(-0.01, 0.01, 3)
    #combine data and labels to shuffle
    trainingSet = np.append(trainingSet,trainingLabels,axis=1)
    #mix the training set up.
    np.random.shuffle(trainingSet)
    iterations = 0
    currentIteration = 0
    gradientNorm = 1
    entropys = []
    norms = []
    print("Starting online training with weight = " +str(rate))
    while(iterations < 100000 and gradientNorm > .001):
        error = np.zeros(3)
        entropy = 0
        output =  outputFunction(trainingSet[currentIteration],weights)
        error = gradient(trainingSet[currentIteration,:-1],trainingSet[currentIteration,-1],output)
        entropy = crossEntropy(output,trainingSet[currentIteration,-1])
        #update weights
        entropys.append(entropy)
        weights += (-rate * error)
        #update loop variables
        iterations += 1
        currentIteration += 1
        if(currentIteration >= len(trainingSet)): currentIteration = 0
   
            
        gradientNorm = np.linalg.norm(error,ord=1)
        norms.append(gradientNorm)


    fig, axs = plt.subplots(1,2)

    subTitle="Entropy vs Iteration"
    axs[0].set_title(subTitle)
    axs[0].plot(entropys)
   
    subTitle="Gradient Norm vs Iteration"
    axs[1].set_title(subTitle)
    axs[1].plot(norms)
    plt.suptitle('Training Rate = ' + str(rate))
    imageTitle = "online_figures"+str(rate)+".png"
    plt.savefig(imageTitle)
    fig.clear()
    print("Training with rate = "+str(rate)+" took " +str(iterations) +" iterations")
    return weights

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

#Define training Rates that will be used to evaluate the model
rates = [1,0.1,0.01,0.001]


#Run the batch training for each learning rate

for i in rates:
    weights = batch(trainingSet,trainingLabels,i)
    batchAccuracy = prediction(weights,testingSet,testingLabels)

    print("Accuracy: " + str(batchAccuracy))
    slope = weights[0]/weights[1]
    intercept = weights[2]/weights[1]
    plt.scatter(testingSet0[:,0],testingSet0[:,1])
    plt.scatter(testingSet1[:,0],testingSet1[:,1])
    plt.plot(testingSet[:,0], (-slope*testingSet[:,0] - intercept),'r')
    plt.title("Batch Model, Learning Rate: " + str(i) + "Accuracy:"+str(batchAccuracy))
    plt.tight_layout()
    batchModel = "batch_model" + str(i) +".png"
    plt.savefig(batchModel)


for i in rates:
    weights = online(trainingSet,trainingLabels,i)
    onlineAccuracy = prediction(weights,testingSet,testingLabels)
    print("Accuracy: " + str(onlineAccuracy))
    slope = weights[0]/weights[1]
    intercept = weights[2]/weights[1]
    plt.scatter(testingSet0[:,0],testingSet0[:,1])
    plt.scatter(testingSet1[:,0],testingSet1[:,1])
    plt.plot(testingSet[:,0], (-slope*testingSet[:,0] - intercept),'r')
    plt.title("Online Model, Learning Rate: " + str(i) + "Accuracy:"+str(onlineAccuracy))
    plt.tight_layout()
    onlineModel = "online_model" + str(i) + ".png"
    plt.savefig(onlineModel)
    

#onlineWeights = online(trainingSet,trainingLabels,.1)