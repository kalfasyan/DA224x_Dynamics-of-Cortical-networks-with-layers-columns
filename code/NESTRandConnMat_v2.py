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

def is_periodic(samples, tolerance=0):
    diffs = [f-g for f,g in zip(samples,samples[1:])]
    return all(d-tolerance <= np.mean(diffs) <= d+tolerance for d in diffs)

def LambertWm1(x):
    nest.sli_push(x); nest.sli_run('LambertWm1'); y=nest.sli_pop()
    return y

def ComputePSPnorm(tauMem, CMem, tauSyn):
  """Compute the maximum of postsynaptic potential
     for a synaptic input current of unit amplitude
     (1 pA)"""

  a = (tauMem / tauSyn)
  b = (1.0 / tauSyn - 1.0 / tauMem)

  # time of maximum
  t_max = 1.0/b * ( -LambertWm1(-exp(-1.0/a)/a) - 1.0/a )

  # maximum of PSP for current of unit amplitude
  return exp(1.0)/(tauSyn*CMem*b) * ((exp(-t_max/tauMem) - exp(-t_max/tauSyn)) / b - t_max*exp(-t_max/tauSyn))


file_name = sys.argv[4]
conmat = np.load("eigens/"+file_name+".dat")
prefix = file_name+"_"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]

startbuild= time.time()

dt      = 0.1    # the resolution in ms
simtime = 1000.0 # Simulation time in ms
delay   = 1.5    # synaptic delay in ms

# Parameters for asynchronous irregular firing
g       = 7.5	 # default 5.0
eta     = 2.0
epsilon = 0.23    # connection probability

order     = 145
NE        = len(pm.exc_nrns_set)
NI        = len(pm.inh_nrns_set)
N_neurons = NE+NI
N_rec     = 50 # record from 50 neurons

CE    = epsilon*NE   # number of excitatory synapses per neuron
CI    = epsilon*NI   # number of inhibitory synapses per neuron
C_tot = int(CI+CE)  # total number of synapses per neuron

# Initialize the parameters of the integrate and fire neuron
tauSyn = 0.5
CMem = 250.0
tauMem = 20.0*3.0**(10/10.0)
theta  = 20.0
J      = 0.1 # postsynaptic amplitude in mV

# normalize synaptic current so that amplitude of a PSP is J
J_ex  = J / ComputePSPnorm(tauMem, CMem, tauSyn)
J_in  = -g*J_ex

# threshold rate, equivalent rate of events needed to
# have mean input current equal to threshold
nu_th  = (theta * CMem) / (J_ex*CE*numpy.exp(1)*tauMem*tauSyn)
nu_ex  = eta*nu_th
p_rate = 1000.0*nu_ex*CE*1.5

nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.1,'local_num_threads':1,'overwrite_files':True,'data_path':'./data/','data_prefix':prefix})

neuron_params= {"C_m"       : CMem,
                "tau_m"     : tauMem,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref"     : 2.0,
                "E_L"       : 0.0,
                "V_reset"   : 0.0,
                "V_m"       : 0.0,
                "V_th"      : theta}

nest.SetDefaults("iaf_psc_alpha", neuron_params)
Nestrons = nest.Create('iaf_psc_alpha',pm.nrns)

#sys.argv[1]  12.3/1.5    -3.5*6.
exc_weights = float(sys.argv[1])
inh_weights = float(sys.argv[2])
ext = float(sys.argv[3])
nest.CopyModel("static_synapse","excitatory",{"weight": exc_weights, "delay":1.5})
nest.CopyModel("static_synapse","inh",{"weight": inh_weights,"delay": 1.5})
nest.CopyModel("static_synapse","ext",{"weight": ext, "delay":1.5})

nest.CopyModel("static_synapse","excitatory",{"weight":J_ex, "delay":delay})
nest.CopyModel("static_synapse","inhibitory",{"weight":J_in, "delay":delay})



spikerec = nest.Create('spike_detector',1)# len(pm.split_lr23)+len(pm.split_lr4)+len(pm.split_lr5))
nest.SetStatus(spikerec,{'to_file':True,'to_memory':False})

nest.SetDefaults("poisson_generator",{"rate": p_rate})
psn = nest.Create('poisson_generator', 1, {'rate':9520.965}) #1150

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
pl.figure(prefix)
pl.plot(xx[:,1],xx[:,0],'.')
pl.title(prefix)
#pm.save("./figures/"+prefix, ext="png", close=False, verbose=True)
pl.savefig('./figures/'+prefix+".", bbox_inches='tight')
#pl.show()