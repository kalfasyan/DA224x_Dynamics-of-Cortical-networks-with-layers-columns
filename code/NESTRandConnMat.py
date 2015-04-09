import nest
nest.ResetKernel()
nest.SetKernelStatus({'print_time':True,'local_num_threads':4})
import parameters as pm
import numpy as np
import matplotlib.pylab as pl
import itertools
import os
import random
from progressbar import *
import RandomConnMat as rm
import time
import pyspike as spk

# Progress bar stuff
# --------------------------------------------------------
widgets = ['Connection Matrix: ', Percentage(), ' ', Bar(marker='=',
            left='[',right=']'), ' ', ETA(), ' ', FileTransferSpeed()]
pbar = ProgressBar(widgets=widgets, maxval=pm.nrns)
#---------------------------------------------------------
start_time = time.time()


def n_connect(j,i):#(p,j,i):##
    if i in pm.exc_nrns_set:
        return (nest.Connect(pm.Nestrons[j],pm.Nestrons[i],syn_spec="excitatory"))# if random.random() < p else 0
    elif i in pm.inh_nrns_set:
        return (nest.Connect(pm.Nestrons[j],pm.Nestrons[i],syn_spec="inhibitory"))# if random.random() < p else 0


nest.CopyModel("static_synapse","excitatory",{"weight":48., "delay":0.5})
nest.CopyModel("static_synapse","inhibitory",{"weight":-4.,"delay":0.5})


spikerec = nest.Create('spike_detector',1)# len(pm.split_lr23)+len(pm.split_lr4)+len(pm.split_lr5))

voltm = nest.Create('voltmeter', 1)
psn = nest.Create('poisson_generator', pm.nrns, {'rate':5000.})

conn_dict = {'rule': 'fixed_indegree', 'indegree': pm.nrns}


for i in range(len(psn)):
    nest.Connect(psn[i],pm.Nestrons[i][0], syn_spec="excitatory")
    nest.Connect(pm.Nestrons[i][0],spikerec[0])


print "Done first phase!"
""" 4. Connection Matrix operations """
conn_matrix = np.zeros((pm.nrns,pm.nrns))
print "created empty matrix!"
pbar.start()

#"""
for j in range(pm.nrns):
    for i in range(pm.nrns):
        if (rm.conn_matrix[j][i] > 0.0001 or rm.conn_matrix[j][i] < -0.0001):
            n_connect(j,i)
    pbar.update(j)
pbar.finish()
"""
for j in range(pm.nrns):
    for i in range(pm.nrns):
        #print j
        if pm.same_hypercolumn(i,j):
            if pm.same_minicolumn(i,j):
                # LAYER 2/3
                if i in pm.layers23 and j in pm.layers23:
                    if pm.both_exc(i,j):
                        n_connect(0.26,i,j)
                    elif pm.both_inh(i,j):
                        n_connect(0.25,i,j)
                    elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                        n_connect(0.21,i,j)
                    else:
                        n_connect(0.16,i,j)
                # LAYER 4
                elif i in pm.layers4 and j in pm.layers4:
                    if pm.both_exc(i,j):
                        n_connect(0.17,i,j)
                    elif pm.both_inh(i,j):
                        n_connect(0.50,i,j)
                    elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                        n_connect(0.19,i,j)
                    else:
                        n_connect(0.10,i,j)
                # LAYER 5
                elif i in pm.layers5 and j in pm.layers5:
                    if pm.both_exc(i,j):
                        n_connect(0.09,i,j)
                    elif pm.both_inh(i,j):
                        n_connect(0.60,i,j)
                    elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                        n_connect(0.10,i,j)
                    else:
                        n_connect(0.12,i,j)
                # FROM LAYER4 -> LAYER2/3
                elif i in pm.layers4 and j in pm.layers23:
                    if pm.both_exc(i,j):
                        n_connect(0.28,i,j)
                    elif pm.both_inh(i,j):
                        n_connect(0.20,i,j)
                    elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                        n_connect(0.10,i,j)
                    else:
                        n_connect(0.50,i,j)
                # FROM LAYER2/3 -> LAYER5
                elif i in pm.layers23 and j in pm.layers5:
                    if pm.both_exc(i,j):
                        n_connect(0.55,i,j)
                    elif pm.both_inh(i,j):
                        n_connect(0.0001,i,j)
                    elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                        n_connect(0.001,i,j)
                    else:
                        n_connect(0.20,i,j)
            # DIFFERENT MINICOLUMN
            elif pm.same_hypercolumn(i,j) and not pm.same_minicolumn(i,j):
                n_connect(0.3,i,j)
        # DIFFERENT HYPERCOLUMN
        elif not pm.same_hypercolumn(i,j):
            n_connect(0.3,i,j)
        else:
            pass
        #pbar.update(i)
#pbar.finish()
#"""
print "Created matrix!"
print ("Matrix Created in %.5s seconds." % (time.time() - start_time))
"""" 5. Connectivity list for weight figure """
#"""
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
a = nest.GetConnections()

nest.Simulate(100.)

data = nest.GetStatus(spikerec)
a,b = data[0]['events']['times'],data[0]['events']['senders']

output = open('PySpike_testdata.txt', 'wb')
z,x =[],[]
for i in range(len(a)):
    if b[i] == b[0]:
        z.append(a[i])
    if b[i] == b[100]:
        x.append(a[i])

diffs = [f-g for f,g in zip(z,z[1:])]

for item in z:
  output.write("%s " % item)
output.write("\n")
for item in x:
  output.write("%s " % item)

diffs = [(z[i+1] - val) for i, val in enumerate(z) if i<len(z)-1]
pl.plot(diffs)
pl.show()


"""
kolor=['c','k','g','b','y','m','c','k','g','b','y','m','y','g','k','b','c','m']
for i in xrange(len(spikerec)):
    a,b = data[i]['events']['times'],data[i]['events']['senders']
    pl.scatter(a,b,marker='.')#,color=kolor[i])
pl.show()
"""
print "Done..!"