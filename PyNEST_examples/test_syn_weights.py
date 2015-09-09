
'''
coments
'''
import nest
import numpy as np
import pylab as pl

nest.ResetKernel()

neuron_params = {'V_th':-55.0, 'V_reset': -70.0, 't_ref': 2.0, 'tau_m':16.6,'C_m':200.0, 'tau_syn_ex':0.3,'tau_syn_in': 0.3,'E_L' : -70.}

nest.CopyModel("static_synapse","excitatory",{"weight":8., "delay":1.})
nest.CopyModel("static_synapse","inhibitory",{"weight":-8., "delay":1.})
conn_dict = {'rule': 'fixed_indegree', 'indegree': 10} # TEST

# create neuron and multimeter
n = nest.Create('iaf_psc_exp', 1,neuron_params)

m = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.1, 'record_from': ['V_m']})#, 'g_ex', 'g_in']})

poi = nest.Create('poisson_generator',1,{'rate':250000.})
# Create spike generators and connect
gex = nest.Create('spike_generator', params = {'spike_times': np.array([500.0, ])})
gin = nest.Create('spike_generator',  params = {'spike_times': np.array([400.0])})

nest.Connect(gex, n, conn_dict) # excitatory
nest.Connect(gin, n, conn_dict) # excitatory
#nest.Connect(poi, n, params={'weight': 1.6}) # inhibitory
nest.Connect(m, n)

# simulate
nest.Simulate(1500)

# obtain and display data
events = nest.GetStatus(m)[0]['events']
t = events['times'];
xx = events['V_m']
epsp = np.max(xx)-xx[0]
ipsp = xx[0] - np.min(xx)

print epsp
print ipsp

pl.figure(1)
pl.clf()
pl.plot(t, events['V_m'])
pl.ylabel('Membrane potential [mV]')
pl.xlabel('Time [msec]')

pl.show()

