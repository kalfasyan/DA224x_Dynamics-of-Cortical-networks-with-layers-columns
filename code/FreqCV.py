import numpy as np
import matplotlib.pylab as py
import parameters_v1 as pm
from NeuroTools import signals

prefix = '111_1._-18._1.8_5000.'
sd_filename = './data/'+prefix+'spike_detector-2881-0.gdf'
data_file = signals.NestFile(sd_filename, with_time = True)
idlist = range(0,pm.nrns) #[i for i in (pm.layers4) if i in pm.split_hc[0]]
spikes = signals.load_spikelist(data_file, t_start=100., t_stop=2000., dims = 1, id_list = idlist)

print "mean ISI: ",np.nanmean(spikes.cv_isi())
print "Mean rate: ",spikes.mean_rate()

bins = 5.

a = np.mean(spikes.firing_rate(5.), axis=0)
fano = np.var(a)/np.mean(a)
print "Fano factor: ", fano
py.figure(111)
py.subplot(411)
py.title(prefix+"\n mean rates")
py.plot(spikes.mean_rates())
py.subplot(412)
py.title("cv isi")
py.plot(spikes.cv_isi())
py.subplot(413)
py.plot(spikes.cv_isi(),spikes.mean_rates(),'.')
spikes.spike_histogram(bins,display=py.subplot(414),normalized=True)
py.show()
"""

 x = analysis.crosscorrelate(spikes[118],spikes[119],display=True)


prefix = '000_19_-144_19'
xx = np.loadtxt('./data/'+prefix+'spike_detector-721-0.gdf')

print "Firing rate: %.2f" % (len(xx[:,0])/2000.*1000./720.)

n_id = np.unique(xx[:,0])
#n_id = np.unique(xx[:,1])

fr_h = np.histogram(xx[:,0],n_id)

py.figure(3)
py.plot(fr_h[0])
py.title("spiking frequency")
#py.show()


ed = np.linspace(0,720,100)
pop_hh = np.histogram(xx[:,1],ed)
py.figure(4)
py.plot(ed[0:-1],pop_hh[0])
py.title("spiking histogram")

xt = xx[xx[:,0]==1,1]
cv_isi = np.zeros(len(n_id))
for ii in range(len(n_id)):
    xt = xx[xx[:,0]==n_id[ii],1]
    isi = np.diff(xt)
    cv_isi[ii] = np.std(isi)/np.mean(isi)

py.figure(5)
py.plot(cv_isi)
py.title("cv isi")
py.show()





xx = np.loadtxt(sd_filename)
no_neurons = 2880 #pm.nrns
bin_size = 5.
ed = np.arange(0,no_neurons,bin_size)
pop_x = np.histogram(xx[:,1],ed)
pop_hh = pop_x[0]/(bin_size*1e-3)/no_neurons

ff = np.var(pop_hh)/np.mean(pop_hh)
print "Fano factor:", ff
"""