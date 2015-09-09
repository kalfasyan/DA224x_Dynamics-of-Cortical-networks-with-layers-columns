import nest
import numpy as np
import os
import simulation_parameters_MN as sp
import pylab as plt
import nest.raster_plot
PS = sp.global_parameters()
params = PS.params

nest.ResetKernel()

neuron = nest.Create("iaf_cond_exp",3)

vm = nest.Create("voltmeter",3)
nest.SetStatus(vm,{"withtime": True})
nest.Connect([vm[0]],[neuron[0]])
nest.Connect([vm[1]],[neuron[1]])
nest.Connect([vm[2]],[neuron[2]])

sd = nest.Create("spike_detector",3)
nest.Connect([neuron[0]],[sd[0]])
nest.Connect([neuron[1]],[sd[1]])
nest.Connect([neuron[2]],[sd[2]])

w=1.5
d=0.5
pois = nest.Create("poisson_generator",2,{'rate': 3000.})

nest.CopyModel("static_synapse","excitatory",{"weight":7., "delay":0.5})
nest.CopyModel("static_synapse","inhibitory",{"weight":-1., "delay":0.5})
nest.Connect([pois[0]],[neuron[0]], syn_spec="excitatory")
nest.Connect([pois[1]],[neuron[1]], syn_spec="inhibitory")
nest.Connect([pois[0]],[neuron[2]], syn_spec="excitatory")
nest.Connect([pois[1]],[neuron[2]], syn_spec="inhibitory")

nest.Simulate(500.)

data3 = nest.GetStatus([vm[2]])
data2 = nest.GetStatus([vm[0]])
data = nest.GetStatus([vm[1]])
spikes = nest.GetStatus(sd)

print data[0]['n_events']
print data2[0]['n_events']

#plt.plot(data3[0]['events']['times'], data3[0]['events']['V_m'])
plt.plot(data[0]['events']['V_m'], data2[0]['events']['V_m'])
#plt.plot(data2[0]['events']['times'], data2[0]['events']['V_m'])
plt.title("%s spikes excitatory" %spikes[0]['n_events']+"/%s spikes both inh+exc" %spikes[2]['n_events'])
plt.ylabel('Vm (mV)')
plt.xlabel('time (secs)')
plt.show()

"""
nest.raster_plot.from_device(vm, hist=True)
nest.raster_plot.show()
"""