"""
Given a data file name (prefix) which contains modelname_exc_inh_1.8_poisson
the scripts calculates mean ISI, Mean rate and Fano factor using NeuroTools.signals
"""
import numpy as np
import matplotlib.pylab as py
import parameters_v1 as pm
from NeuroTools import signals

prefix = '111_1.1_-17.0_1.8_4700.0'
sd_filename = './data/'+prefix+'spike_detector-2881-0.gdf'
data_file = signals.NestFile(sd_filename, with_time = True)

"""" Choose_Layer(layer, hypercolumn, minicolumn) """
idlist = range(0,pm.nrns)#pm.choose_Layer(pm.layers23,0,0)##
#idlist2 = pm.choose_Layer(pm.layers4,0,0)

""" Collect spikes from t_start to t_stop for neurons in id_list """
spikes = signals.load_spikelist(data_file, t_start=100., t_stop=2000., dims = 1, id_list = idlist)
#spikes2 = signals.load_spikelist(data_file, t_start=100., t_stop=2000.,dims = 1, id_list = idlist2)

""" Calculate 'Population Histogram' and 'Fano Factor' """
bins = 5.
xx = np.loadtxt(sd_filename)
ed = np.arange(100,2e+3,bins)
pop_x = np.histogram(xx[:,1],ed)
pop_hh = pop_x[0]/(bins*1e-3)/pm.nrns
ff = np.var(pop_hh)/np.mean(pop_hh)

print "mean ISI: ",np.nanmean(spikes.cv_isi())
print "Mean rate: ", np.mean(pop_hh)
print "Fano factor: ", ff

""" Plotting: raster, mean-rates, cv_isi, histogram """
py.figure(1)
py.title("raster plot")
spikes.raster_plot(display=py.subplot(511))
py.subplot(512)
py.title("mean rates")
py.ylabel("Hrz")
py.plot(spikes.mean_rates())
py.subplot(513)
py.title("cv isi")
py.ylabel("ms")
py.plot(spikes.cv_isi())
py.subplot(514)
py.plot(spikes.cv_isi(),spikes.mean_rates(),'.')
py.subplot(515)
#spikes.spike_histogram(bins,display=py.subplot(515),normalized=True)
py.plot(ed[0:-1],pop_x[0])
py.show()