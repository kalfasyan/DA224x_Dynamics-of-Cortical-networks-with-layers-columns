import nest
import nest.raster_plot
import nest.voltage_trace
import parameters_v1 as pm
import numpy as np
import matplotlib.pylab as pl
import sys
import time
import bokeh.plotting as bk


""" Functions for Bokeh plotting """
#-----------------------------------------------------------------------------
def mscatter(p, x, y, typestr):
    p.scatter(x, y, marker=typestr,
            line_color="#6666ee", fill_color="#ee6666", fill_alpha=0.5, size=12)
def mtext(p, x, y, textstr):
    p.text(x, y, text=textstr,
         text_color="#449944", text_align="center", text_font_size="10pt")
#-----------------------------------------------------------------------------

file_name = sys.argv[1]
conmat = np.load("eigens/"+file_name+".dat")
prefix = file_name+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads':1,'overwrite_files':True,'data_path':'./data/','data_prefix':prefix})
start_time = time.time()


""" PARAMETERS """
#conn_dict = {'rule': 'fixed_indegree', 'indegree': pm.nrns}
#conn_dict2 = {'rule': 'fixed_outdegree', 'outdegree': 1}
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

xx = np.loadtxt('./data/'+prefix+'spike_detector-2881-0.gdf')

pl.figure(prefix)
#nest.raster_plot.from_device(spikerec, hist=True)
pl.plot(xx[:,1],xx[:,0],'.')
#pl.title(prefix)
#pl.savefig('./figures/'+prefix+".", bbox_inches='tight')
pl.show()




"""
bk.output_file("prefix.html")
N = pm.nrns
x = xx[:,1]
y = xx[:,0]
TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,save,hover"

p = bk.figure(tools=TOOLS)
mscatter(p, x,y,"asterisk")
bk.save(p, filename="./figures/"+prefix+"html")
bk.show(p)
"""

"""
 for i in range(0,3):
    mple = [xx[j,:] for j in pm.split_hc[i]]
    ed = np.arange(1000,2e+3,5)
    pop_xist = np.histogram(mple,ed)
    figure(555),plot(ed[0:-1],pop_xist[0])
"""