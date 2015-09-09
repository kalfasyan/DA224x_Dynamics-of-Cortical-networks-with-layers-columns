import nest
import numpy as np
import os
import simulation_parameters_MN as sp
import pylab as pl
import nest.raster_plot
PS = sp.global_parameters()
params = PS.params


# create the minicolumns
mcs = [ None for i in xrange(params['n_mc'])]
for i_mc in xrange(params['n_mc']):
   mcs[i_mc] = nest.Create(params['neuron_type'], params['n_exc_per_mc'])

########################################################################

# create inhibitory neurons for cross inhibition
inh_pops = [ None for i in xrange(params['n_hc'])]
for i_hc in xrange(params['n_hc']):
    inh_pops[i_hc] = nest.Create(params['neuron_type'], params['n_inh_per_hc'])

# create spike detector
spikerec = nest.Create('spike_detector', params['n_mc'])

# create input source to minicolumns
poiss = nest.Create('poisson_generator', params['n_mc'],{'rate': 4000.})

########################################################################
# connect cells within a minicolumns
n_tgt = int(np.round(params['p_ee_local'] * params['n_exc_per_mc']))
for i_mc in xrange(params['n_mc']):
    nest.RandomConvergentConnect(mcs[i_mc], mcs[i_mc], n_tgt, \
            weight=params['w_ee_local'], delay=params['delay_ee_local'], \
            options={'allow_autapses': False, 'allow_multapses': False})
    nest.RandomConvergentConnect(mcs[i_mc], inh_pops[0], n_tgt, \
            weight=params['w_ee_local'], delay=params['delay_ee_local'], \
            options={'allow_autapses': False, 'allow_multapses': False})
    nest.DivergentConnect(poiss, mcs[i_mc], params['w_ei_unspec']*i_mc/2,params['delay_ee_global'])
    nest.ConvergentConnect(mcs[i_mc], spikerec)


# check the connectivity
connections = nest.GetConnections()
conn_list = np.zeros((len(connections), 3)) # 3 columns for src, tgt, weight
#print 'example connections:'
#for i in connections:
#	print i
#print 'information from nest.GetStatus', nest.GetStatus([connections[0]])

for i_ in xrange(len(connections)):
    info = nest.GetStatus([connections[i_]])
    weight = info[0]['weight']
    conn_list[i_, 0] = connections[i_][0]
    conn_list[i_, 1] = connections[i_][1]
    conn_list[i_, 2] = weight


np.savetxt('debug_connectivity.txt', conn_list)

nest.Simulate(100.0)

data = nest.GetStatus(spikerec)
nest.raster_plot.from_device(spikerec, hist=True)
nest.raster_plot.show()


