'''
A simple TensorFlow network with two  convolutional layers and two fully connected layers.  The network is trained to classify the MNIST dataset AND output the weights and biases so they can be run on a non-TensorFlow environment (e.g., CUDNN, Python, etc.).  
This program is a good test to see that you have your file formats correct.
Author: Alex Terrazas, PhD
Project: https://github.com/DrEvil1963/EmbeddedDeepNets.git
'''

from __future__ import print_function

import tensorflow as tf
import timeit
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNISTDATA/", one_hot=True)

# Parameters
learning_rate = 0.001
training_batch_size = 128
n_batches = 20000
n_pixels = 784 # Input Pixels (28*28=784)
n_classes = 10 # Output Classes (0-9 digits)
dropout = 0.75 # Drop 25% of units on each iteration to prevent overtraining
row_major = 0 #C++ and Python store arrays in Row Major and Row Minor
screen_update = 55 #update the screen every n iterations

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_pixels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
 
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
 
    # Convolution Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, 2)

    # Convolution Layer 1 
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max Pooling (down-sampling)
  
    conv2 = maxpool2d(conv2, 4)
  
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
   
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['bout'])
    return out

def train_net(optimizer, idx):
    
     batch_train, batch_truth = mnist.train.next_batch(training_batch_size)

     # Determine the backprop signal    
     sess.run(optimizer, feed_dict={x: batch_train, y: batch_truth, 
                                        keep_prob: dropout})
          
     if idx % screen_update == 0:
     # Output loss and accuracy for this batch to update the screen
       loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_train,
                                                         y: batch_truth,
                                                         keep_prob: 1.})
       
       print("Iteration: %10u  Training Accuracy: %10f  Percent Done: %5f" % (idx,100.*acc,100.*idx/n_batches))


def writeFile(inputarray, fname):
   if row_major == 1:
      inputarray = np.swapaxes(inputarray, 0, 1)

   tmp = inputarray.astype(dtype=np.float32)
      
   outbytes=bytearray(tmp)
   newFile = open(fname, "wb")
   newFile.write(outbytes)
   newFile.close();


# Store layers weight & bias
wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20],stddev=.1,  name="wc1"))
wc2 = tf.Variable(tf.truncated_normal([5, 5, 20, 50],stddev=.1, name="wc2"))
wd1 = tf.Variable(tf.truncated_normal([4*4*50, 500],stddev=.1, name="wd1"))
wout = tf.Variable(tf.truncated_normal([500, n_classes],stddev=.1))

weights = {
    'wc1': wc1,
    'wc2': wc2,
    'wd1': wd1,
    'out': wout
}

bc1 = tf.Variable(tf.truncated_normal([20],stddev=0.1), name="bc1")
bc2 = tf.Variable(tf.truncated_normal([50],stddev=0.1),name="bc2")
bd1 = tf.Variable(tf.truncated_normal([500],stddev=0.1),name="bd1")
bout = tf.Variable(tf.truncated_normal([n_classes],stddev=0.1))

biases = {
    'bc1': bc1,
    'bc2': bc2,
    'bd1': bd1,
    'bout': bout
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

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
    #Loop over training batches

    # Keep training until reach max iterations
    while indx < n_batches:
        train_net(optimizer, indx)
        indx = indx + 1

    print("Training Completed.")
  
    accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.})
    print("TEST ACCURACY ON 256 MNIST Test Sample: ", accuracy * 100)
   
    # Get the biases and write them to bin files
    # The names are 
    bc1, bc2, bd1, bout = sess.run([bc1,bc2,bd1,bout])
    writeFile(bc1, "./conv1.bias.bin") 
    writeFile(bc2, "./conv2.bias.bin")     
    writeFile(bd1, "./fclayer1.bias.bin")
    writeFile(bout, "./fclayer2.bias.bin")
 
    wc1, wc2, wd1, wout = sess.run([wc1,wc2,wd1,wout])
    writeFile(wc1, "./conv1.bin")
    writeFile(wc2, "./conv2.bin")
    writeFile(wd1, "./fclayer1.bin")
    writeFile(wout, "./fclayer2.bin")
