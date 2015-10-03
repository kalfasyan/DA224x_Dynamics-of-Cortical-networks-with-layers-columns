"""
Create plots for Synchrony (Fano Factors) and Regularity (Mean firing frequency)
for all eight models
"""

import numpy as np
#import pylab as py
from time import sleep
import sys
import pickle
import pylab as py

def switch_modelname(s):
    if s == '000':
        return 'R'
    elif s == '111':
        return 'LMH'
    elif s == '101':
        return 'LH'
    elif s == '100':
        return 'L'
    elif s == '110':
        return 'LM'
    elif s == '011':
        return 'MH'
    elif s == '010':
        return 'M'
    elif s == '001':
        return 'H'
    else:
        print 'Wrong name'

total_neuron = 2880
fanopic,freqpic = open('fano_pickle.dmp','w'),open('freq_pickle.dmp','w')


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

print "\nDone!"
# saving ff_all and fr_all
# you can save them with specific names, and load them later
# so you won't have to create them again
ff_all.dump('./dumps/ff_all.dat')
fr_all.dump('./dumps/fr_all.dat')

# Initializing Figures
fig,ax = py.subplots()
fig = py.gcf()

# Creating lists of labels for the axes
lblinh = [str(i)[:3] for i in inh]
lblnu = [str(i)[:2] for i in nu]

for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        p_id =  ii*8 + jj
        py.subplot(8,8,p_id+1)
        if p_id in range(0,8):
            py.title("exc. weight: "+str(exc[jj]))
        if jj in [0,8,16,24,32,40,48,56,64]:
            py.ylabel(switch_modelname(f_prefix[ii]))
        yy = ff_all[ii,jj,:,:]
        py.imshow(yy, interpolation='nearest', aspect='auto')#,cmap = py.cm.Blues)
        #py.title(str(p_id))
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(yy.shape[0]), minor=False)
        ax.set_xticks(np.arange(yy.shape[1]), minor=False)
        ax.set_xticklabels(lblnu, minor=False)
        ax.set_yticklabels(lblinh,minor=False,)
        #py.clim(0,3)
        py.colorbar()
        ax.grid(False)
        ax = py.gca()
ax.set_frame_on(False)
ax.set_yticks(np.arange(yy.shape[0]), minor=False)
ax.set_xticks(np.arange(yy.shape[1]), minor=False)
ax.set_xticklabels(lblnu, minor=False)
ax.set_yticklabels(lblinh,minor=False)
#py.colorbar()
pickle.dump(ax, fanopic)
fanopic.close()

fig,ax = py.subplots()
fig = py.gcf()

for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        p_id =  ii*8 + jj
        py.subplot(8,8,p_id+1)
        if p_id in range(0,8):
            py.title("exc. weight: "+str(exc[jj]))
        if jj in [0,8,16,24,32,40,48,56,64]:
            py.ylabel(switch_modelname(f_prefix[ii]))
        yy = fr_all[ii,jj,:,:]
        py.imshow(yy, interpolation='nearest', aspect='auto')#,cmap = py.cm.Blues)
        #py.title(str(p_id))
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(yy.shape[0]), minor=False)
        ax.set_xticks(np.arange(yy.shape[1]), minor=False)
        ax.set_xticklabels(lblnu, minor=False)
        ax.set_yticklabels(lblinh,minor=False,)
        #py.clim(0,6)
        py.colorbar()
        ax.grid(False)
        ax = py.gca()
ax.set_frame_on(False)
ax.set_yticks(np.arange(yy.shape[0]), minor=False)
ax.set_xticks(np.arange(yy.shape[1]), minor=False)
ax.set_xticklabels(lblnu, minor=False)
ax.set_yticklabels(lblinh,minor=False)
#py.colorbar()
pickle.dump(ax, freqpic)
freqpic.close()


# this opens the pickle and shows the plot
bx = pickle.load(file('./dumps/freq_pickle.dmp'))

#py.figure(3)
#py.clf()
#for ii in range(len(f_prefix)):
#    frx = np.reshape(fr_all[ii,:,:,:],100)
#    ffx = np.reshape(ff_all[ii,:,:,:],100)
#    py.plot(frx,ffx,'.')
#py.show()