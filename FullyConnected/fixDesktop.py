import numpy as np
import math
import matplotlib.pyplot as pt
import struct

#fix the ip1 weights
with open("ip1D.bin", "rb") as f:
    bytes = f.read(313600)
    L1w = struct.unpack('78400f', bytes)

    testme=np.insert(L1w, 0, L1w[78399])
    L1wFixed=np.delete(testme, 78400)
    f.close()

wd1=np.ones([78400], dtype=np.float32)
wd1[:] = L1wFixed[:]
outbytes=bytearray(wd1)
newFile = open("./ip1FIXED.bin", "wb")
newFile.write(outbytes)
newFile.close()

#now do ip2 weights
with open("ip2D.bin", "rb") as f:
    bytes = f.read(4000)
    L2w = struct.unpack('1000f', bytes)

    testme=np.insert(L2w, 0, L2w[999])
    L2wFixed=np.delete(testme, 1000)
    f.close()

wd2=np.ones([1000], dtype=np.float32)
wd2[:] = L2wFixed[:]
outbytes=bytearray(wd2)
newFile = open("./ip2FIXED.bin", "wb")
newFile.write(outbytes)
newFile.close()

#ip1 biases
with open("ip1D.bias.bin", "rb") as f:
    bytes = f.read(400)
    L1b = struct.unpack('100f', bytes)
    testme=np.insert(L1b, 0, L1b[99])
    L1bFixed=np.delete(testme, 100)
    f.close()

wb1=np.ones([100], dtype=np.float32)
wb1[:] = L1bFixed[:]
outbytes=bytearray(wb1)
newFile = open("./ip1FIXED.bias.bin", "wb")
newFile.write(outbytes)
newFile.close()


#ip2 biases
with open("ip2D.bias.bin", "rb") as f:
    bytes = f.read(40)
    L2b = struct.unpack('10f', bytes)
    testme=np.insert(L2b, 0, L2b[9])
    L2bFixed=np.delete(testme, 10)
    f.close()

wb2=np.ones([10], dtype=np.float32)
wb2[:] = L2bFixed[:]
outbytes=bytearray(wb2)
newFile = open("./ip2FIXED.bias.bin", "wb")
newFile.write(outbytes)
newFile.close()
