import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

#function for plotting the 
def plotting(weights2):
  zero = np.reshape(weights2[:,0],(28,28))
  one = np.reshape(weights2[:,1],(28,28))
  two = np.reshape(weights2[:,2],(28,28))
  three = np.reshape(weights2[:,3],(28,28))
  four = np.reshape(weights2[:,4],(28,28))
  five = np.reshape(weights2[:,5],(28,28))
  six = np.reshape(weights2[:,6],(28,28))
  seven = np.reshape(weights2[:,7],(28,28))
  eight = np.reshape(weights2[:,8],(28,28))
  nine = np.reshape(weights2[:,9],(28,28))
  fig,axs = plt.subplots(2,5)
  axs[0,0].imshow(one)
  axs[0,1].imshow(two)
  axs[0,2].imshow(three)
  axs[0,3].imshow(four)
  axs[0,4].imshow(five)
  axs[1,0].imshow(six)
  axs[1,1].imshow(seven)
  axs[1,2].imshow(eight)
  axs[1,3].imshow(nine)
  axs[1,4].imshow(zero)
  plt.tight_layout
  plt.show()

#obtain training and testing data from the mnist dataset
#data is 28x28 images of hand drawn digits. 
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  #Creates the input layer. Turns the input data into a vector
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #creates a hidden layer with 512 nodes
  #uses rectified linear unit as the activation function
  #tf.keras.layers.Dense(512, activation=tf.nn.relu),
  #applies drop out to the input data. Will turn 20% of inputs into 0s
  tf.keras.layers.Dropout(0.2),
  #Creates the output layer with 10 nodes using Softmax for activation function
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#creates the model with the layers specified above, also specifies metrics to be reported
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model. epochs is the number of iterations
model.fit(x_train, y_train, epochs=5)
#check the model with the testing data
print("Evaluation:")
model.evaluate(x_test, y_test)

weights = model.get_weights()
weights = np.array(weights)
weights2 = np.array(weights[0])
plotting(weights2)
