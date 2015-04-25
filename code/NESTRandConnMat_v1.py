import nest
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
#import pylab as py
from NeuroTools import signals


file_name = sys.argv[3]
conmat = np.load("eigens/"+file_name+".dat")
prefix = file_name+"_"+sys.argv[1]+'_'+sys.argv[2]+"_"
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads':1,'overwrite_files':True,'data_path':'./data/','data_prefix':prefix})

# Progress bar stuff
# --------------------------------------------------------
widgets = ['Connection Matrix: ', Percentage(), ' ', Bar(marker='=',
            left='[',right=']'), ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=pm.nrns)
#---------------------------------------------------------
start_time = time.time()

def is_periodic(samples, tolerance=0):
    diffs = [f-g for f,g in zip(samples,samples[1:])]
    return all(d-tolerance <= np.mean(diffs) <= d+tolerance for d in diffs)



def n_connect(j,i):#(p,j,i):##
    if i in pm.exc_nrns_set:
        return (nest.Connect(pm.Nestrons[j],pm.Nestrons[i],syn_spec="excitatory"))# if random.random() < p else 0
    elif i in pm.inh_nrns_set:
        return (nest.Connect(pm.Nestrons[j],pm.Nestrons[i],syn_spec="inhibitory"))# if random.random() < p else 0

#sys.argv[1]  12.3/1.5    -3.5*6.
exc_weights = float(sys.argv[1])
inh_weights = float(sys.argv[2])

nest.CopyModel("static_synapse","excitatory",{"weight": exc_weights, "delay":1.5})
nest.CopyModel("static_synapse","ext",{"weight": 12.3, "delay":1.5})
nest.CopyModel("static_synapse","inh",{"weight": inh_weights,"delay": 1.5})


Nestrons = nest.Create('iaf_cond_exp',pm.nrns)#,params={'I_e':350.})

spikerec = nest.Create('spike_detector',1)# len(pm.split_lr23)+len(pm.split_lr4)+len(pm.split_lr5))
nest.SetStatus(spikerec,{'to_file':True,'to_memory':False})
voltm = nest.Create('voltmeter', 1)
psn = nest.Create('poisson_generator', 1, {'rate':1150.})

#psn_stim1 = nest.Create('poisson_generator', 1, {'rate':1000.,'start':300.,'stop':350.})
#psn_stim2 = nest.Create('poisson_generator', 1, {'rate':1000.,'start':800.,'stop':850.})
#psn_stim3 = nest.Create('poisson_generator', 1, {'rate':1000.,'start':1200.,'stop':1250.})

conn_dict = {'rule': 'fixed_indegree', 'indegree': pm.nrns}
conn_dict2 = {'rule': 'fixed_outdegree', 'outdegree': 1}

nid = np.zeros(pm.nrns)
nid[:] = Nestrons
pbar.start()
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
    pbar.update(j)
pbar.finish()



nest.Connect(psn,Nestrons, syn_spec="ext")
#nest.Connect(psn_stim1,Nestrons[0:30], syn_spec="ext")
#nest.Connect(psn_stim2,Nestrons[0:30], syn_spec="ext")
#nest.Connect(psn_stim3,Nestrons[90:120], syn_spec="ext")

nest.Connect(Nestrons,spikerec)

print "Done first phase.."

#"""



print ("Matrix Created in %.5s seconds.." % (time.time() - start_time))
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
"""
xx = np.loadtxt('./data/'+prefix+'spike_detector-721-0.gdf')

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
#pl.figure(998)
#pl.plot(xx[:,1],xx[:,0],'.')
#pl.show()

