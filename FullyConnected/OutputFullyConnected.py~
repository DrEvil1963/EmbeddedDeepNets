'''
A simple TensorFlow fully connected neural network to classify the MNIST dataset AND output the weights and biases so they can be run on a non-TensorFlow environment (e.g., CUDNN, Python, etc.).  
This program is a good test to see that you have your file formats correct.
Author: Alex Terrazas, PhD
Project: https://github.com/DrEvil1963/EmbeddedDeepNets.git
'''

from __future__ import print_function

import tensorflow as tf
import timeit
import numpy as np
import math

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Use a permanent directory to store the data
mnist = input_data.read_data_sets("/home/drevil/Downloads/MNINST/", one_hot=True)

# Parameters
learning_rate = 0.001
training_batch_size = 128
n_batches = 20000
n_pixels = 784 # Input Pixels (28*28=784)
n_classes = 10 # Output Classes (0-9 digits)
dropout = 0.75 # Drop 25% of units on each iteration to prevent overtraining
row_major = 0 #C++ and Python store arrays in Row Major and Row Minor

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_pixels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# xavier initialization (improves accuracy substantially)
def xavier_init(n_inputs, n_outputs):
    stddev = math.sqrt(6.0 / (n_inputs + n_outputs))
    return stddev

# Create model
def simple_net(x, weights, biases, dropout):
 
    # Reshape the input (x) to fit fully connected layer input
    fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # Use the rectified linear unit activation function on the first fully connected layer
    fc1 = tf.nn.relu(fc1)
  
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['bout'])
    return out

def train_net(optimizer, printcode, idx):
    
     batch_train, batch_truth = mnist.train.next_batch(training_batch_size)

     # Determine the backprop signal    
     sess.run(optimizer, feed_dict={x: batch_train, y: batch_truth, 
                                        keep_prob: dropout})
      
     if printcode == 0:
     # Calculate batch loss and accuracy
       loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_train,
                                                         y: batch_truth,
                                                         keep_prob: 1.})
       
       print("Iteration: " , str(idx), "Training Accuracy: ", acc*100)


def writeFile(inputarray, fname):
   if row_major == 1:
      inputarray = np.swapaxes(inputarray, 0, 1)

   tmp = inputarray.astype(dtype=np.float32)
      
   outbytes=bytearray(tmp)
   newFile = open(fname, "wb")
   newFile.write(outbytes)
   newFile.close();


# Declare weights & biases for the two-layer network
wd1 = tf.Variable(tf.truncated_normal([n_pixels, 100],
                                      stddev=xavier_init(n_pixels,100)))
wout = tf.Variable(tf.truncated_normal([100, n_classes],
                                       stddev=xavier_init(100,n_classes)))

bd1 = tf.Variable(tf.truncated_normal([100],stddev=1.0))
bout = tf.Variable(tf.truncated_normal([n_classes],stddev=1.0))

weights = {
    'wd1': wd1,
    'out': wout
}


biases = {
    'bd1': bd1,
    'bout': bout
}


# Construct model
pred = simple_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    indx = 0
    # Loop over training batches
    while indx < n_batches:
       # Print results every 2000 batches
       if indx % 2000 == 0:
          print_flag = 0
       
       train_net(optimizer, print_flag, indx)
       print_flag = 1
       indx = indx + 1 
    
    print("Training Completed...")

    # Run the test images to get accuracy on hold out sample 

    accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.})
    
    print("*******TEST ACCURACY ON 256 MNIST Test Sample: ", accuracy * 100)
    
    # Get the biases and write them to bin files
  
    bd1, bout = sess.run([bd1,bout])
    writeFile(bd1, "./ip1D.bias.bin")
    writeFile(bout, "./ip2D.bias.bin")

    # Get the weights and write them to bin files
    wd1, wout = sess.run([wd1,wout])
    writeFile(wd1, "./ip1D.bin")
    writeFile(wout, "./ip2D.bin")

