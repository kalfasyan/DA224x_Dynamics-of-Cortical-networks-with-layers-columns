import nest
import nest.raster_plot
import nest.voltage_trace
import parameters_v1 as pm
import numpy as np
import matplotlib.pylab as pl
import itertools
import os
import sys
import random
import time
import pyspike as spk
import types


file_name = sys.argv[1]
conmat = np.load("eigens/"+file_name+".dat")
#prefix = file_name+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]
prefix = 'dokimi'
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads':1,'overwrite_files':True,'data_path':'./data/','data_prefix':prefix})
start_time = time.time()




""" PARAMETERS """
#conn_dict = {'rule': 'fixed_indegree', 'indegree': pm.nrns}
exc = float(sys.argv[2])
inh = float(sys.argv[3])
ext = float(sys.argv[4])
p_rate = float(sys.argv[5])
nest.CopyModel("static_synapse","excitatory",{"weight": exc, "delay":1.5})
nest.CopyModel("static_synapse","inh",{"weight": inh,"delay": 1.5})
nest.CopyModel("static_synapse","ext",{"weight": ext, "delay":1.5})
neuron_params = {'V_th':-55.0, 'V_reset': -70.0, 't_ref': 2.0, 'g_L':16.6,'C_m':250.0, 'E_ex': 0.0, 'E_in': -80.0, 'tau_syn_ex':0.2,'tau_syn_in': 2.0,'E_L' : -70.}
nest.SetDefaults("iaf_cond_alpha", neuron_params)




""" NEST CREATIONS """
Nestrons = nest.Create('iaf_cond_alpha',pm.nrns)#,params={'I_e':350.})
spikerec = nest.Create('spike_detector',1)# len(pm.split_lr23)+len(pm.split_lr4)+len(pm.split_lr5))
nest.SetStatus(spikerec,{'to_file':True,'to_memory':False, 'start':100.})
psn = nest.Create('poisson_generator', 1, {'rate':p_rate}) #1150
psn1 = nest.Create('poisson_generator', 1, {'rate': p_rate/5.})
x=np.zeros(6)
hc,mc = 0,0
x[0],x[1] = pm.choose_EI(pm.layers23,hc,mc)
x[2],x[3] = pm.choose_EI(pm.layers4,hc,mc)
x[4],x[5] = pm.choose_EI(pm.layers5,hc,mc)
vm = nest.Create('voltmeter',len(x))
nest.SetStatus(vm,{'to_file':True,'to_memory':False})
for i in range(len(x)):
    nest.SetStatus([Nestrons[int(x[i])]],{'V_th':1000.})
    nest.Connect([vm[i]],[Nestrons[int(x[i])]])




""" CONNECTIONS """
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
nest.Connect(Nestrons,spikerec)



""" SIMULATION AND PLOTTING """
print "Simulating.."
nest.Simulate(2000.)
print "Done! Now plotting..."

#xx = np.loadtxt('./data/'+prefix+'spike_detector-2881-0.gdf')
vv = [np.loadtxt('./data/'+prefix+'voltmeter-288'+str(i)+'-0.dat') for i in range(4,10)]

pl.figure(prefix)
for i in range(len(vv)):
    pl.subplot(len(vv),1,i)
    pl.title("neuron "+str(int(x[i]))+","+pm.n_where(x[i],conmat))
    pl.plot(vv[i][:,1], vv[i][:,2])
#pl.plot(vv[len(vv)][:,1],vv[len(vv)][:,2])
    #nest.voltage_trace.from_device([vm[i]])
#pl.plot(vv[:,1], vv[:,2])
#nest.raster_plot.from_device(spikerec, hist=True)
#pl.plot(xx[:,1],xx[:,0],'.')
#pl.title(prefix)
#pl.savefig('./figures/'+prefix+".", bbox_inches='tight')
pl.show()