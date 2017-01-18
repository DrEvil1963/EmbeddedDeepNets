Embedded Deep Nets teaching material shows how to export weights from a TensorFlow model to run a 'forward pass' on an embedded device --such as the Jetson TX1 Develop Kit.  

There are three example directories:  FullyConnected, ConvOneLayer, and ConvTwoLayer

Inside each directory are two python programs:  

1) an Output program (OutputFullyConnected.py, OutputConvOneLayer.py and OutputConvTwoLayer.py) 

2) a Run program (RunFullyConnected.py. RunConvOneLayer.py and RunConvTwoLayer.py)

Each time you run the Output program you will generate weights and bias files that can be used to run a model.  For example, if you type $OutputFullyConnected.py, you will generate two weights files and two bias files--fclayer1.bin, fclayer2.bin, fclayer1.bias.bin and fclayer2.bias.bin.  

Once you have trained a network and generated the weights and biases files, you can run the "forward pass".  So, for the FullyConnected network, you would type $RunFullyConnected.py.  You will then get a plot of the input image and the network's guess about what that image represents.



 # EmbeddedDeepNets
