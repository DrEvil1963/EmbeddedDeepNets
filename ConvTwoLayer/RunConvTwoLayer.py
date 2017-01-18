import struct
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import glob

def simple_relu(x):
    out = np.maximum(0,x)
    return out

def plotImage(mess, img):
    print(mess)
    fig = plt.figure()
    plt.imshow(img, cmap = "gray")
    plt.show()
 
def maxPool(imgin,outdim, ksize):
   outmax=np.zeros([outdim, outdim], dtype=np.float32)
   for x in xrange(outdim):
      for y in xrange(outdim):
       rw = x*ksize
       cl = y*ksize
       outmax[x,y] = np.max(imgin[rw:rw+ksize-1, cl:(cl+ksize-1)])
   #plotImage("outmax: ", outmax)

   return outmax


def runConv(weighttensor, bias, sampleimage, nchans, nfilters, outdim, ksize):  
   outputimages = np.zeros([outdim,outdim,nfilters], dtype=np.float32)
   for filt in xrange(nfilters):
     outsum=np.zeros([sampleimage.shape[0], sampleimage.shape[1]], dtype=np.float32)
     for chan in xrange(nchans):
     
       w = weighttensor[:,:,chan,filt]
       gr=sg.correlate2d(sampleimage[:,:,chan],w, boundary='symm', mode='same')
       
       outsum=np.add(outsum,gr)
     outsum=np.add(outsum, bias[filt])
     outputimages[:,:,filt] = maxPool(outsum/nfilters, outdim, ksize) 
     
   return simple_relu(outputimages)

with open("conv1.bin", "rb") as f:
   c1data=struct.unpack('500f', f.read(2000))
   testc1=np.reshape(c1data, (5,5,1,20))
f.close()
with open("conv1.bias.bin", "rb") as f:
   c1bias = struct.unpack('20f',f.read(80))
f.close()

with open("conv2.bin", "rb") as f:
   c2data = struct.unpack('25000f', f.read(100000))
   testc2=np.reshape(c2data, (5,5,20,50))
f.close()
with open("conv2.bias.bin", "rb") as f:
   c2bias = struct.unpack('50f',f.read(200))
f.close()

with open("fclayer1.bin", "rb") as f:
   fc1data = struct.unpack('400000f', f.read(1600000))
f.close()
with open("fclayer1.bias.bin", "rb") as f:
   fc1bias = struct.unpack('500f', f.read(2000))
f.close()
with open("fclayer2.bin", "rb") as f:
    fc2data = struct.unpack('5000f', f.read(20000))
f.close()
with open("fclayer2.bias.bin", "rb") as f:
   fc2bias = struct.unpack('10f', f.read(40))
f.close()

imfiles = glob.glob("../MNISTDATA/*.pgm")

for x in xrange(20):

  with open(imfiles[x], "rb") as f:
    hdr = f.read(52)
    imdata=struct.unpack('784B', f.read(784))
    testim=np.zeros([28,28,1])
    testim[:,:,0]=np.reshape(imdata, (28,28))
    f.close()
  conv1output = runConv(testc1,c1bias, testim, 1 ,20, 14, 2)
  conv2output = runConv(testc2, c2bias, conv1output, 20 ,50, 4, 4)
  flatc2 = conv2output.flatten()
  L1_output=np.add(np.matmul(flatc2,np.reshape(fc1data, (800,500))),fc1bias)
  L2_output=np.add(np.matmul(simple_relu(L1_output),np.reshape(fc2data, (500,10))), fc2bias)

  print(L2_output)
  print("RESULT: ", np.argmax(L2_output))
  plotImage("",testim[:,:,0])
