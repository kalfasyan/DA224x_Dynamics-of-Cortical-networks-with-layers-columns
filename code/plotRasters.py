"""
Creates a Raster plot for the 'prefix' parameters

Reminder for parameters:
modelname_exc.weight_inh.weight_1.8_poissonRate-spike_detector-2881-0.gdf
example:
111_1.3_-26.0_1.8_4400.0spike_detector-2881-0.gdf
"""

import sys
import numpy as np
import matplotlib.pyplot as py

modelname = sys.argv[1]
figurename = sys.argv[2]
prefix = modelname+'_1.3_-26.0_1.8_4400.0spike_detector-2881-0.gdf'
xx = np.loadtxt('./data/'+prefix)

py.figure(figurename)
#nest.raster_plot.from_device(spikerec, hist=True)
py.plot(xx[:,1],xx[:,0],'b.')
py.title(figurename)
py.savefig(figurename+'.')
py.xlabel('Time (ms)')
py.ylabel('Neuron ID')
py.gcf().subplots_adjust(bottom=0.15)
#py.gcf().tight_layout()
py.show()
