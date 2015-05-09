
'''
coments
'''
import nest
import numpy as np
import pylab as pl

nest.ResetKernel()
#J_e = 1.2 # == 1.3mV at -70
#J_i = -0.725 # == 0.45mV at -55
#J_i = -1.135 # == -0.7mV at -55

neuron_params = {
'V_th':-0.0, 'V_reset': -70.0, 't_ref': 5.0, 'g_L':10.0,'C_m':200.0, 'E_ex': 0.0, 'E_in': -80.0, 'tau_syn_ex':5.0,'tau_syn_in': 10.0,'tau_minus':20.0,'E_L' : -70.}
neuron_params = {
'V_th':-55.0, 'V_reset': -70.0, 't_ref': 2.0, 'g_L':16.7,'C_m':250.0, 'E_ex': 0.0, 'E_in': -80.0, 'tau_syn_ex':0.33,'tau_syn_in': 0.33,'E_L' : -70.}

neuron_params = {'V_th':-55.0, 'V_reset': -70.0, 't_ref': 2.0, 'g_L':16.7,'C_m':250.0, 'E_ex': 0.0, 'E_in': -80.0, 'tau_syn_ex':0.33,'tau_syn_in': 0.33,'E_L' : -70.}
#neuron_params = {'V_th':-45.0, 'V_reset': -80.0, 't_ref': 2.0, 'g_L':12.5,'C_m':200.0, 'E_ex': 0.0, 'E_in': -80.0, 'tau_syn_ex':0.3,'tau_syn_in': 2.,'E_L' : -46.}

neuron_params = {'V_th':-54.0, 'V_reset': -70.0, 't_ref': 2.0, 'g_L':16.6,'C_m':250.0, 'E_ex': 0.0, 'E_in': -80.0, 'tau_syn_ex':0.2,'tau_syn_in': 2.0,'E_L' : -70.}

# display recordables for illustration
print 'iaf_cond_alpha recordables: ', nest.GetDefaults('iaf_cond_alpha')['recordables']

# create neuron and multimeter
n = nest.Create('iaf_cond_alpha', 1,neuron_params)

m = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.1, 'record_from': ['V_m', 'g_ex', 'g_in']})

poi = nest.Create('poisson_generator',1,{'rate':6000.})
# Create spike generators and connect
gex = nest.Create('spike_generator', params = {'spike_times': np.array([500.0, ])})
gin = nest.Create('spike_generator',  params = {'spike_times': np.array([400.0])})


nest.CopyModel("static_synapse","exc",{"weight":1.4, "delay":1.5})
nest.CopyModel("static_synapse","inh",{"weight":-0.43, "delay":1.5})

nest.Connect(gex, n, syn_spec="exc") # excitatory
nest.Connect(gin, n, syn_spec="inh") # inhibitory
nest.Connect(poi, n, syn_spec='exc') # inhibitory
nest.Connect(m, n)

# simulate
nest.Simulate(2000)

# obtain and display data
events = nest.GetStatus(m)[0]['events']
t = events['times'];
xx = events['V_m']
rej_id = 2000
epsp = np.max(xx[rej_id:-1])-xx[rej_id]
ipsp = xx[rej_id] - np.min(xx[rej_id:-1])

print epsp
print ipsp

pl.figure(1)
pl.clf()
pl.subplot(211)
pl.plot(t, events['V_m'])
#pl.axis([190, 250, -56.7, -55])
pl.ylabel('Membrane potential [mV]')

pl.subplot(212)
pl.plot(t, events['g_ex'], t, events['g_in'])
#pl.axis([0, 100, -55.55, -55.])
pl.xlabel('Time [ms]')
pl.ylabel('Synaptic conductance [nS]')
pl.legend(('g_exc', 'g_inh'))
pl.show()

