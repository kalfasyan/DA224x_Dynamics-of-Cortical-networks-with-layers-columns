import numpy as np
import pylab as py
import parameters_v1 as pm
from scipy.stats.stats import pearsonr
import sys

#orig_stdout = sys.stdout
f = file('corrOutput.txt', 'a')
#sys.stdout = f


hc = int(sys.argv[1])
mc = int(sys.argv[2])
lr1 = sys.argv[3]
lr2 = sys.argv[4]

lrdict = {'23': pm.layers23,
          '4': pm.layers4,
          '5': pm.layers5}

prefix = '111_1._-18._1.8_5000.'
sd_filename = './data/'+prefix+'spike_detector-2881-0.gdf'



xx = np.loadtxt(sd_filename)
n_id = xx[:,0]
ed = np.arange(100,2e+3,5.)

cor1 = np.zeros((len(xx[:,:]),2))
for i in range(len(xx[:,1])):
    if n_id[i] in lrdict[lr1] and n_id[i] in pm.split_mc[hc][mc]:
        cor1[i] = xx[:,:][i]
hist_cor1 = np.histogram(cor1[:,1],ed)

cor2 = np.zeros((len(xx[:,:]),2))
for i in range(len(xx[:,1])):
    if n_id[i] in lrdict[lr2] and n_id[i] in pm.split_mc[hc][mc]:
        cor2[i] = xx[:,:][i]
hist_cor2 = np.histogram(cor2[:,1],ed)

print hc,mc,lr1,lr2,pearsonr(hist_cor1[0],hist_cor2[0])[0]


#sys.stdout = orig_stdout
#f.close()
#py.plot(ed[0:-1],hist_cor1[0])
#py.plot(ed[0:-1],hist_cor2[0])
#py.show()