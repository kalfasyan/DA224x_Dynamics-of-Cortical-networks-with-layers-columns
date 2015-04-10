import nest
import parameters_v1 as pm
import numpy as np
import matplotlib.pylab as pl
import itertools
import os
import random
from progressbar import *
import RandomConnMat as rm
import time
import pyspike as spk
import pylab as py

file_name = 'test'

nest.ResetKernel()
nest.SetKernelStatus({'print_time':True,'local_num_threads':4,'overwrite_files':True,'data_path':'./data/','data_prefix':file_name})

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


nest.CopyModel("static_synapse","excitatory",{"weight":18., "delay":1.5})
nest.CopyModel("static_synapse","inhibitory",{"weight":-4.,"delay":1.5})


Nestrons = nest.Create('iaf_cond_exp',pm.nrns)

spikerec = nest.Create('spike_detector',1)# len(pm.split_lr23)+len(pm.split_lr4)+len(pm.split_lr5))
nest.SetStatus(spikerec,{'to_file':True,'to_memory':False})
voltm = nest.Create('voltmeter', 1)
psn = nest.Create('poisson_generator', 1, {'rate':5000.})

conn_dict = {'rule': 'fixed_indegree', 'indegree': pm.nrns}
conn_dict2 = {'rule': 'fixed_outdegree', 'outdegree': 1}

nid = np.zeros(pm.nrns)
nid[:] = Nestrons
pbar.start()
for j in range(pm.nrns):
    post_id = py.find(np.abs(rm.conn_matrix[:,j])>0.0001)
    post_wt = rm.conn_matrix.T[j][post_id]
    post_del = np.ones(np.size(post_wt))*1.
    nx = nid[post_id].astype(int).tolist()
    if j in pm.exc_nrns_set:
        syn_type = 'excitatory'
    elif j in pm.inh_nrns_set:
        syn_type = 'inhibitory'
    nest.Connect([Nestrons[j]],nx,conn_dict2,syn_spec=syn_type)
    pbar.update(j)
pbar.finish()



nest.Connect(psn,Nestrons, syn_spec="excitatory")

nest.Connect(Nestrons,spikerec,conn_dict)

print "Done first phase!"
print "created empty matrix!"

#"""


print "Created matrix!"
print ("Matrix Created in %.5s seconds." % (time.time() - start_time))
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
cons = nest.GetConnections()

nest.Simulate(100.)



output = open('PySpike_testdata.txt', 'wb')
"""
z,x =[],[]
# a's are the spike times of a particular b (sender)
for i in range(len(a)):
    if b[i] == b[0]:
        z.append(a[i])
    if b[i] == b[100]:
        x.append(a[i])


for item in z:
  output.write("%s " % item)
output.write("\n")
for item in x:
  output.write("%s " % item)

delta=[[] for i in range(len(a))]
for i in range(len(a)):
    for j in range(len(b)):
        if b[j] == b[i]:
            delta[i].append(a[j])

for i in delta:
    output.write("%s \n" % i)
"""

print "Done..! now plotting..."

xx = np.loadtxt('./data/testspike_detector-181-3.gdf')
pl.plot(xx[:,1],xx[:,0],'.')
pl.show()


"""


data = nest.GetStatus(spikerec)
a,b = data[0]['events']['times'],data[0]['events']['senders']

print data[0]['events']['times']
print data[0]['events']['senders']

kolor=['c','k','g','b','y','m','c','k','g','b','y','m','y','g','k','b','c','m']
for i in xrange(len(spikerec)):
    a,b = data[i]['events']['times'],data[i]['events']['senders']
    pl.scatter(a,b,marker='.')#,color=kolor[i])
pl.show()
"""