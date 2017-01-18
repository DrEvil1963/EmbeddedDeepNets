
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import glob

def simple_relu(x):
    out = np.maximum(0,x)
    return out
  
##input image 784
def load_image(fname):
 with open(fname, "rb") as f:
   f.read(52)
   bytes = f.read(784)
   a = struct.unpack('784B', bytes)
   b = np.array(a)
   inp_image = np.divide(b,255.)
   f.close()
   plt.imshow(b.reshape(28,28),cmap="gray")
   return inp_image

def forward_pass(inp_image):

 #input L1 weights 784,100
 with open("ip1D.bin", "rb") as f:
   bytes = f.read(313600)
   L1w = struct.unpack('78400f', bytes)
   L1wA = np.reshape(L1w, (784,100))
   f.close()

 #input L1 biases 100
 with open("ip1D.bias.bin", "rb") as f:
    bytes = f.read(400)
    L1b= struct.unpack('100f', bytes)
    f.close()

 #input L2 weights (100,10)
 with open("ip2D.bin", "rb") as f:
   bytes = f.read(4000)
   L2w = struct.unpack('1000f', bytes)
   L2wA = np.reshape(L2w, (100,10))
   f.close()

 #input L2 biases 10
 with open("ip2D.bias.bin", "rb") as f:
    bytes = f.read(40)
    L2b= struct.unpack('10f', bytes)
    f.close()


# multiply inputs 784 x L1 weights 784x100 = 100 + biases
 L2_input=np.add(np.matmul(inp_image, L1wA), L1b)

#multiply L2_inputs 100 by L2w = 100x10 = 10 + biases
 L2_output=np.add(np.matmul(simple_relu(L2_input), L2wA), L2b)

 print("L2_output: ", L2_output, "  ***ARGMAX: ", np.argmax(L2_output))

imfiles = glob.glob("/home/drevil/Downloads/TFGPUExample/MNISTDATA/*.pgm")
for x in xrange(20):

  with open(imfiles[x], "rb") as f:
    forward_pass(load_image(imfiles[x]))
    plt.show()
 
