import numpy as np
import matplotlib.pylab as py
from NeuroTools import signals

sd_filename = './data/111_19_-144_19spike_detector-721-0.gdf'
data_file = signals.NestFile(sd_filename, with_time = True)
spikes = signals.load_spikelist(data_file, dims = 1, id_list = range(0,720))
py.subplot(211)
py.title("mean rates")
py.plot(spikes.mean_rates())
py.subplot(212)
py.title("cv isi")
py.plot(spikes.cv_isi())
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
"""