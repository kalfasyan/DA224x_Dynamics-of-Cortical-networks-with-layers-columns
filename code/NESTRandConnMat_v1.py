import nest
import nest.raster_plot
import parameters_v1 as pm
import numpy as np
import matplotlib.pylab as pl
import itertools
import os
import sys
import random
from progressbar import *
#import RandomConnMat as rm
import time
import pyspike as spk
import types
#import nest.
#import pylab as py


file_name = sys.argv[4]
conmat = np.load("eigens/"+file_name+".dat")
prefix = file_name+"_"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads':1,'overwrite_files':True,'data_path':'./data/','data_prefix':prefix})


start_time = time.time()

def is_periodic(samples, tolerance=0):
    diffs = [f-g for f,g in zip(samples,samples[1:])]
    return all(d-tolerance <= np.mean(diffs) <= d+tolerance for d in diffs)

exc_weights = float(sys.argv[1])
inh_weights = float(sys.argv[2])
ext = float(sys.argv[3])
nest.CopyModel("static_synapse","excitatory",{"weight": exc_weights, "delay":1.5})
nest.CopyModel("static_synapse","inh",{"weight": inh_weights,"delay": 1.5})
nest.CopyModel("static_synapse","ext",{"weight": ext, "delay":1.5})


neuron_params= {"C_m"       : 250.0,
                "tau_m"     : 20.0*3.0**(10/10.0),
                "tau_syn_ex": 0.5,
                "tau_syn_in": 0.5,
                "t_ref"     : 2.0,
                "E_L"       : 0.0,
                "V_reset"   : 0.0,
                "V_m"       : 0.0,
                "V_th"      : 20.0}

nest.SetDefaults("iaf_psc_alpha", neuron_params)
Nestrons = nest.Create('iaf_psc_alpha',pm.nrns)#,params={'I_e':350.})

spikerec = nest.Create('spike_detector',1)# len(pm.split_lr23)+len(pm.split_lr4)+len(pm.split_lr5))
nest.SetStatus(spikerec,{'to_file':True,'to_memory':False})
voltm = nest.Create('voltmeter', 1)
psn = nest.Create('poisson_generator', 1, {'rate':9520.965}) #1150

#psn_stim1 = nest.Create('poisson_generator', 1, {'rate':1000.,'start':300.,'stop':350.})
#psn_stim2 = nest.Create('poisson_generator', 1, {'rate':1000.,'start':800.,'stop':850.})
#psn_stim3 = nest.Create('poisson_generator', 1, {'rate':1000.,'start':1200.,'stop':1250.})

conn_dict = {'rule': 'fixed_indegree', 'indegree': pm.nrns}
conn_dict2 = {'rule': 'fixed_outdegree', 'outdegree': 1}

nid = np.zeros(pm.nrns)
nid[:] = Nestrons
for j in range(pm.nrns):
    post_id = pl.find(np.abs(conmat[:,j]) != 0.) #>0.0001)
    post_wt = conmat.T[j][post_id]
    post_del = np.ones(np.size(post_wt))*1.
    nx = nid[post_id].astype(int).tolist()
    if j in pm.exc_nrns_set:
        syn_type = 'excitatory'
    elif j in pm.inh_nrns_set:
        syn_type = 'inh'
    nest.Connect([Nestrons[j]],nx,'all_to_all',syn_spec=syn_type)

nest.Connect(psn,Nestrons, syn_spec="ext")
#nest.Connect(psn_stim1,Nestrons[0:30], syn_spec="ext")
#nest.Connect(psn_stim2,Nestrons[0:30], syn_spec="ext")
#nest.Connect(psn_stim3,Nestrons[90:120], syn_spec="ext")
nest.Connect(Nestrons,spikerec)

print "Done first phase.."

"""" 5. Connectivity list for weight figure """
"""
connections = nest.GetConnections()
conn_list = np.zeros((len(connections), 3))
for i_ in xrange(len(connections)):
    info = nest.GetStatus([connections[i_]])
    weight = info[0]['weight']
    conn_list[i_, 0] = connections[i_][0]
    conn_list[i_, 1] = connections[i_][1]
    conn_list[i_, 2] = weight

np.savetxt('debug_connectivity.txt', conn_list)
nest.PrintNetwork()
#"""
#cons = nest.GetConnections()


nest.Simulate(2000.)



print "Done..\n now plotting..."

xx = np.loadtxt('./data/'+prefix+'spike_detector-721-0.gdf')

"""
output = open('PySpike_testdata.txt', 'wb')
delta = [[] for i in range(pm.nrns)]
for i in range(pm.nrns):
    searchval = i
    q = np.where(xx[:,0]==searchval)[0]
    for j in q:
        delta[i].append(xx[j,1])
for i in range(pm.nrns):
    output.write("%s \n" %delta[i])
"""

events = nest.GetStatus(spikerec,"n_events")[0]
print events/2000.*1000./720.

pl.figure(prefix)
#nest.raster_plot.from_device(spikerec, hist=True)
pl.plot(xx[:,1],xx[:,0],'.')
pl.title(prefix)
pl.savefig('./figures/'+prefix+".", bbox_inches='tight')
pl.show()