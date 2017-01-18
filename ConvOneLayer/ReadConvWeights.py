import struct
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt


with open("conv1.bin", "rb") as f:
   c1data=struct.unpack('500f', f.read(2000))
   testc1=np.reshape(c1data, (5,5,1,20))
f.close()

print("0,0:", testc1[:,:,0,0])
print("0,1:", testc1[:,:,0,1])
