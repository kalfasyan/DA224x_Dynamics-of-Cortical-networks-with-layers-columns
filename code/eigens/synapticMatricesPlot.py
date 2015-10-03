"""
Plots the synaptic matrix of a given model
Example:
run synapticMatricesPlot.py 101 LH
"""

import matplotlib.pyplot as py
import numpy as np
import sys

modelname = sys.argv[1]
figurename = sys.argv[2]
c1 = np.load(modelname+'.dat')
py.imshow(c1, interpolation='none',origin='lower')
py.title(figurename)
py.xlabel("Neuron ID")
py.ylabel("Neuron ID")
py.savefig(figurename+'.png',bbox_inches='tight')
#py.show()