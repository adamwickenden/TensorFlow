# tensorflow imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# image processing imports
import numpy as np
from PIL import Image

import os

#MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # y labels are oh-encoded

#DATA Totals
n_train = mnist.train.num_examples
n_validation = mnist.validation.num_examples
n_test = mnist.test.num_examples

#No nodes per layer
n_input = 784 #input - 28x28 pixels
n_hidden1 = 1568
n_hidden2 = 784
n_hidden3 = 128
n_output = 10 #output - 0-9 digits

#Initial set parameters
learning_rate = 1e-4 # Amount by which the parameters (node weights) may adjust per training pass
n_iterations = 10000 # n of iterations per training iteration
batch_size = 128 # n of iterations per batch within each iteration
dropout = 0.5 # dropout of 50% to be used for final hidden layer to prevent overfitting

#Placeholder Tensors
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)


#Weightings and biases are the values which determine the strength of the connections between
#each of the nodes between layers. These are the values that are altered by the neural net
#Initial weightings - random values linking the nodes of each layer
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

#Initial biases - small values are used initially to ensure propagation initiates
biases = {
    'b1': tf.Variable(tf.constant(0.05, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.05, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.05, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.05, shape=[n_output]))
}

#Layer operations. Between each layer, we perform matrix multiplication with the
#outputs of the previous layer and the current layers weights and then add the bias
#to the weights. The final hidden layer uses a dropout operation w/ P(0.5)
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

#optimisation algorithms. Cross entromy with gradient descent optimisation
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
    )
)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Define evaluation methods to measure how well the network is being trained
# compare predictions (output) to true values (Y) and return an array of
# booleans, the Number of which can be averaged to give an accuracy value
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initialise session for training/testing
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# we aim to reduce the loss in the optimisation function (difference between prediction/labels)
#this is done by propagating forward (predicting) then comparing the results (compute loss) and
# propagating backward (altering the weights etc)

#train in mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
    })
    # print loss and accuracy = per minibatch not overall, so wont follow the expected decrease/increase
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy),
        )

# Now we test against the test images, with a a keep_prob=1
# so all final hidden layer nodes are used

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set: ", test_accuracy)

# import image, convert RGBA to grayscale, invert 0-255 values (black-white), flatten array with ravel
img1 = np.invert(Image.open("./tensorflow-demo/test/1.png").convert('L')).ravel()
img2 = np.invert(Image.open("./tensorflow-demo/test/2.png").convert('L')).ravel()
img3 = np.invert(Image.open("./tensorflow-demo/test/3.png").convert('L')).ravel()
img5 = np.invert(Image.open("./tensorflow-demo/test/5.png").convert('L')).ravel()
img6 = np.invert(Image.open("./tensorflow-demo/test/6.png").convert('L')).ravel()
img7 = np.invert(Image.open("./tensorflow-demo/test/7.png").convert('L')).ravel()

# Now we can run a session against a single image instead of against a test set
# then print out the Y (label) value

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img1]})
print ("Prediction for test image 1: ", np.squeeze(prediction))

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img2]})
print ("Prediction for test image 2: ", np.squeeze(prediction))

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img3]})
print ("Prediction for test image 3: ", np.squeeze(prediction))

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img5]})
print ("Prediction for test image 5: ", np.squeeze(prediction))

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img6]})
print ("Prediction for test image 6: ", np.squeeze(prediction))

prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img7]})
print ("Prediction for test image 7: ", np.squeeze(prediction))