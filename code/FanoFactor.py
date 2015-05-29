"""
python FanoFactor.py inhFiles 111 > fano/fanoInh100.txt
python FanoFactor.py excFiles 111 > fano/fanoExc100.txt
python FanoFactor.py extFiles 111 > fano/fanoExt100.txt
"""
import sys
import parameters_v1 as pm
import glob
import numpy as np
import time
start = time.clock()

mc = [0,1,2,3]
hc = [0,1,2]
layersets,flags = [],[]
for i in hc:
    for j in mc:
       layersets.append(pm.choose_Layer(pm.layers23,i,j))
       flags.append("L23 hc"+str(i)+" mc"+str(j))
       layersets.append(pm.choose_Layer(pm.layers4,i,j))
       flags.append("L4 hc"+str(i)+" mc"+str(j))
       layersets.append(pm.choose_Layer(pm.layers5,i,j))
       flags.append("L5 hc"+str(i)+" mc"+str(j))

fname = sys.argv[2]
inhFiles = glob.glob('./data/'+fname+'*1.0*5000*.gdf')
excFiles = glob.glob('./data/'+fname+'*-18*5000*.gdf')
extFiles = glob.glob('./data/'+fname+'*1.0*-18.*.gdf')

inhFiles.sort()
excFiles.sort()
extFiles.sort()

#print len(inhFiles),len(excFiles),len(extFiles)
#print excFiles,inhFiles,extFiles

user_input = {"inhFiles": inhFiles,
              "excFiles": excFiles,
              "extFiles": extFiles}
filenames = user_input[sys.argv[1]] # User gives e.g inhFiles for inhibitory-increase-data

bins = 5.
ed = np.arange(100,2e+3,bins)
ff,ffnames = [],[]
# This creates ff[i] = fano factors OR/AND ffnames[i] for the labels
for q in filenames:
    sd_filename = q
    #print q[7:]
    xx = np.loadtxt(sd_filename)
    n_id = xx[:,0]
    for i in range(len(layersets)):
        cor1 = np.zeros((len(xx[:,:]),2))
        for j in range(len(xx[:,1])):
            if n_id[j] in layersets[i]:
                cor1[j] = xx[:,:][j]
        pop_x = np.histogram(cor1[:,1],ed)
        pop_hh = pop_x[0]/(bins*1e-3)/pm.nrns
        ff.append( np.var(pop_hh)/np.mean(pop_hh))
        ffnames.append(flags[i])
        print ff[i]

#print "done!"
end = time.clock()
#print "Runtime: %.5s seconds" %(end - start)
