# run correlations.py 2 3 4 5 001

import numpy as np
import parameters_v1 as pm
from scipy.stats.stats import pearsonr
import sys
from pandas import DataFrame as dt

fname = sys.argv[5] #for example: '101'

hc = int(sys.argv[1])
mc = int(sys.argv[2])
lr1 = sys.argv[3]
lr2 = sys.argv[4]

lrdict = {'23': pm.layers23,
          '4': pm.layers4,
          '5': pm.layers5}

prefix = fname+'_1._-18._1.8_5000.'
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

df = dt({'hypercolumn': hc,
         "layerA":lr1,
         'minicolumn': mc,
         "layerB":lr2},
        index=[pearsonr(hist_cor1[0],hist_cor2[0])[0]])

df.to_excel('./test/'+fname+"_"+str(hc)+str(mc)+str(lr1)+str(lr2)+'.xlsx')