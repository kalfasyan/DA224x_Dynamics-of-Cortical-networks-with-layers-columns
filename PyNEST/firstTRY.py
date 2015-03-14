import nest
nest.ResetKernel()
nest.SetKernelStatus({'print_time':True,'local_num_threads':4})
import parameters as pm
import numpy as np
import matplotlib.pylab as pl
import itertools
import os
import random


def flip(p,j,i):
    if i in pm.exc_nrns_set:
        return (nest.Connect(pm.Nestrons[i],pm.Nestrons[j],syn_spec="excitatory")) if random.random() < p else 0
    elif i in pm.inh_nrns_set:
        return (nest.Connect(pm.Nestrons[i],pm.Nestrons[j],syn_spec="inhibitory")) if random.random() < p else 0


nest.CopyModel("static_synapse","excitatory",{"weight":8., "delay":0.5})
nest.CopyModel("static_synapse","inhibitory",{"weight":-4.,"delay":0.5})


spikerec = nest.Create('spike_detector', len(pm.split_lr))
voltm = nest.Create('voltmeter', 1)
psn = nest.Create('poisson_generator', pm.nrns, {'rate':5000.})


for i in range(len(psn)):
    nest.Connect(psn[i],pm.Nestrons[i][0], syn_spec="excitatory")
    #nest.Connect(pm.Nestrons[i][0],spikerec[i])

#nest.ConvergentConnect(pm.Nestrons,spikerec)
#print pm.split_lr

conn_dict = {'rule': 'fixed_indegree', 'indegree': len(pm.split_lr[0][0])}

print pm.split_lr[0][0]
print spikerec[0]


for i in range(len(pm.split_lr)):
    for j in pm.split_lr[i]:
        #print j
        #print spikerec[i]
        nest.Connect(j,[spikerec[i]],conn_dict)

#nest.Connect(pm.split_lr[0][0],[spikerec[0]],conn_dict)

""" 4. Connection Matrix operations """
conn_matrix = np.zeros((pm.nrns,pm.nrns))

for j in range(pm.nrns):
    for i in range(pm.nrns):
        #print j
        if pm.same_hypercolumn(i,j):
              # LAYER 2/3
            if i in pm.layers23 and j in pm.layers23:
                if pm.both_exc(i,j):
                    flip(0.26,i,j)
                elif pm.both_inh(i,j):
                    flip(0.25,i,j)
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    flip(0.21,i,j)
                else:
                    flip(0.16,i,j)
            # LAYER 4
            elif i in pm.layers4 and j in pm.layers4:
                if pm.both_exc(i,j):
                    flip(0.17,i,j)
                elif pm.both_inh(i,j):
                    flip(0.50,i,j)
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    flip(0.19,i,j)
                else:
                    flip(0.10,i,j)
            # LAYER 5
            elif i in pm.layers5 and j in pm.layers5:
                if pm.both_exc(i,j):
                    flip(0.09,i,j)
                elif pm.both_inh(i,j):
                    flip(0.60,i,j)
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    flip(0.10,i,j)
                else:
                    flip(0.12,i,j)
            # FROM LAYER4 -> LAYER2/3
            elif i in pm.layers4 and j in pm.layers23:
                if pm.both_exc(i,j):
                    flip(0.28,i,j)
                elif pm.both_inh(i,j):
                    flip(0.20,i,j)
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    flip(0.10,i,j)
                else:
                    flip(0.50,i,j)
            # FROM LAYER2/3 -> LAYER5
            elif i in pm.layers23 and j in pm.layers5:
                if pm.both_exc(i,j):
                    flip(0.55,i,j)
                elif pm.both_inh(i,j):
                    flip(0.0001,i,j)
                elif i in pm.exc_nrns_set and j in pm.inh_nrns_set:
                    flip(0.001,i,j)
                else:
                    flip(0.20,i,j)
        else:
            if pm.next_hypercolumn(i,j):
                if (i in pm.layers5 and j in pm.layers4):
                    #print "smth"
                    flip(0.35,i,j)
            elif pm.prev_hypercolumn(i,j):
                if (i in pm.layers5 and j in pm.layers23):
                    #print "smth"
                    flip(0.35,i,j)
            else:
                flip(0.00001,i,j)


"""" 5. Connectivity list for weight figure """
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
a = nest.GetConnections()

nest.Simulate(500.)


data = nest.GetStatus(spikerec)
kolor=['c','k','g','b','y','m','c','k','g','b','y','m','y','g','k','b','c','m']
for i in xrange(len(spikerec)):
    a,b = data[i]['events']['times'],data[i]['events']['senders']
    pl.scatter(a,b,marker='.',color=kolor[i])
pl.show()

print "Done..!"