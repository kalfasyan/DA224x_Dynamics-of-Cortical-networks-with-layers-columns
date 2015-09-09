import numpy as np
#import pylab as py
from time import sleep
import sys
import pickle
import pylab as py

total_neuron = 2880
fanopic,freqpic = open('fano_pickle.dmp','w'),open('freq_pickle.dmp','w')

"""
f_prefix = ['000', '111', '101', '100']
exc = [.5, .6, .7, .8, .9]
inh = [-23., -24., -25., -26., -27.]
nu = [4000., 4200., 4400., 4600.]
"""

f_prefix = ['000', '111', '101', '100','110', '011', '010', '001']
exc = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0]
inh = [-18., -20., -22., -24., -26.]
nu = [3000., 3500., 4000., 4400., 4800., 5200., 5600.]

ext_wt = 1.8
dir_name = '/home/yannis/DA224x/code/data/'

bin_size = 2.
ed = np.arange(500.,2e3,bin_size)


ff_all = np.zeros((len(f_prefix),len(exc),len(inh),len(nu)))
fr_all = np.zeros((len(f_prefix),len(exc),len(inh),len(nu)))


for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        for kk in range(len(inh)):
            for mm in range(len(nu)):
                fname = dir_name + f_prefix[ii] + '_' + str(exc[jj]) + '_' + str(inh[kk]) + '_' + str(ext_wt) + '_' + str(nu[mm]) + 'spike_detector-2881-0.gdf'
                #print fname
                xx = np.loadtxt(fname)
                hh,edx = np.histogram(xx[:,1],ed)
                hh_norm = hh/float(total_neuron)/(bin_size*1e-3)
                fr_all[ii,jj,kk,mm] = len(xx[:,1])/total_neuron/1.5
                ff_all[ii,jj,kk,mm] = np.var(hh_norm)/np.mean(hh_norm)
    # Loading Bar
    sys.stdout.write('\r')
    sys.stdout.write("[%-7s] %d%%" % ('='*ii, 14.3*ii))
    sys.stdout.flush()
    sleep(.125)

# saving ff_all and fr_all
ff_all.dump('ff_all.dat')
fr_all.dump('fr_all.dat')
# Initializing Figures
fig,ax = py.subplots()
fig = py.gcf()

# Creating lists of labels for the axes
lblinh = [str(i)[:3] for i in inh]
lblnu = [str(i)[:2] for i in nu]
print "\nDone!"

for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        p_id =  ii*8 + jj
        py.subplot(8,8,p_id+1)
        if p_id in range(0,8):
            py.title("exc: "+str(exc[jj]))
        if jj in [0,8,16,24,32,40,48,56,64]:
            py.ylabel(f_prefix[ii])
        yy = ff_all[ii,jj,:,:]
        #py.title(str(p_id))
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(lblnu, minor=False)
        ax.set_yticklabels(lblinh,minor=False)
        #for y in range(yy.shape[0]):
        #    for x in range(yy.shape[1]):
        #        py.text(x + 0.5, y + 0.5, '%.1f' % yy[y, x],
        #                horizontalalignment='center',
        #                verticalalignment='center',
        #                )
        py.pcolor(yy)#,cmap = py.cm.Blues)
        py.clim(0,6)
        ax.grid(False)
        ax = py.gca()
ax.set_frame_on(False)
ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
ax.set_xticklabels(lblnu, minor=False)
ax.set_yticklabels(lblinh,minor=False)
pickle.dump(ax, fanopic)
fanopic.close()

fig,ax = py.subplots()
for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        p_id =  ii*8 + jj
        py.subplot(8,8,p_id+1)
        if p_id in range(0,8):
            py.title("exc: "+str(exc[jj]))
        if jj in [0,8,16,24,32,40,48,56,64]:
            py.ylabel(f_prefix[ii])
        yy = fr_all[ii,jj,:,:]
        #py.title(str(p_id))
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(lblnu, minor=False)
        ax.set_yticklabels(lblinh,minor=False)
        #for y in range(yy.shape[0]):
        #    for x in range(yy.shape[1]):
        #        py.text(x + 0.5, y + 0.5, '%.1f' % yy[y, x],
        #                horizontalalignment='center',
        #                verticalalignment='center',
        #                )
        py.pcolor(yy)#,cmap = py.cm.Blues)
        py.clim(0,6)
        ax.grid(False)
        ax = py.gca()
ax.set_frame_on(False)
ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
ax.set_xticklabels(lblnu, minor=False)
ax.set_yticklabels(lblinh,minor=False)
pickle.dump(ax, freqpic)
freqpic.close()

"""
py.figure(3)
py.clf()
for ii in range(len(f_prefix)):
    frx = np.reshape(fr_all[ii,:,:,:],100)
    ffx = np.reshape(ff_all[ii,:,:,:],100)
    py.plot(frx,ffx,'.')
py.show()
"""