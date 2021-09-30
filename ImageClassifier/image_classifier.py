import tensorflow as tf
import numpy as np
from tensorflow import keras

# image printing
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_nn_ops import relu

# load the predefined dataset from google
fashion_mist=keras.datasets.fashion_mnist

# pull out images data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mist.load_data()


# making neural link (the input, hidden layer and output)
model=keras.Sequential([
    # input is a 28*28 image ('flattens the 28*28 into a single 784*1 input layer)
    keras.layers.Flatten(input_shape=(28,28)),

    #hidden layer is 128 deep. frelu returns the value, or 0 (works good enough. much faster)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output is 0-10 (depending upon the what piece of cloth it is). returns maximum probability
    keras.layers.Dense(units=10, activation=tf.nn.softmax)

])

# compile our model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

# train our model, using our training data
model.fit(train_images, train_labels, epochs=5)


# test our model, using our testing data
test_loss=model.evaluate(test_images, test_labels)

plt.imshow(test_images[1], cmap='gray', vmin=0, vmax=255)
plt.show()
print(test_labels[1])

# probability of all tested images
predictions=model.predict(test_images)

# list of 10 probablities of second image done for 10 test images
print(predictions[1])

# max probability of second image
print(max(predictions[1]))

# print out max predicted image's label
print(list(predictions[1]).index(max(predictions[1])))
