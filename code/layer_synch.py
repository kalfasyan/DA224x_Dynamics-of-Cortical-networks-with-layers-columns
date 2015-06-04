import numpy as np
import pylab as py

total_neuron = 2880
f_prefix = ['000', '111', '101', '100']
exc = [.5, .6, .7, .8, .9]
inh = [-23., -24., -25., -26., -27.]
nu = [4000., 4200., 4400., 4600.]

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
                print fname
                xx = np.loadtxt(fname)
                hh,edx = np.histogram(xx[:,1],ed)
                hh_norm = hh/float(total_neuron)/(bin_size*1e-3)
                fr_all[ii,jj,kk,mm] = len(xx[:,1])/total_neuron/1.5
                ff_all[ii,jj,kk,mm] = np.var(hh_norm)/np.mean(hh_norm)


fig,ax = py.subplots()
fig = py.gcf()

lblinh = [str(i) for i in inh]
lblnu = [str(i) for i in nu]

for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        p_id =  ii*5 + jj
        py.subplot(4,5,p_id+1)
        yy = ff_all[ii,jj,:,:]
        py.title(str(p_id))
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(lblnu, minor=False)
        ax.set_yticklabels(lblinh,minor=False)
        for y in range(yy.shape[0]):
            for x in range(yy.shape[1]):
                py.text(x + 0.5, y + 0.5, '%.2f' % yy[y, x],
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        py.pcolor(yy)#,cmap = py.cm.Blues)
        py.clim(0,6)
        ax.grid(False)
        ax = py.gca()
ax.set_frame_on(False)
ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
ax.set_xticklabels(lblnu, minor=False)
ax.set_yticklabels(lblinh,minor=False)

fig,ax = py.subplots()
for ii in range(len(f_prefix)):
    for jj in range(len(exc)):
        p_id =  ii*5 + jj +1
        py.subplot(4,5,p_id)
        yy = fr_all[ii,jj,:,:]
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(lblnu, minor=False)
        ax.set_yticklabels(lblinh,minor=False)
        for y in range(yy.shape[0]):
            for x in range(yy.shape[1]):
                py.text(x + 0.5, y + 0.5, '%.2f' % yy[y, x],
                        horizontalalignment='center',
                        verticalalignment='center',
                        )

        py.pcolor(yy)#,cmap = py.cm.Blues)
        py.clim(0,6)
        ax.grid(False)
        ax = py.gca()
ax.set_frame_on(False)
ax.set_yticks(np.arange(yy.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(yy.shape[1]) + 0.5, minor=False)
ax.set_xticklabels(lblnu, minor=False)
ax.set_yticklabels(lblinh,minor=False)

py.figure(3)
py.clf()
for ii in range(len(f_prefix)):
    frx = np.reshape(fr_all[ii,:,:,:],100)
    ffx = np.reshape(ff_all[ii,:,:,:],100)
    py.plot(frx,ffx,'.')
py.show()
