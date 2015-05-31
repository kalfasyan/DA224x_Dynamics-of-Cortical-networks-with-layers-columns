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
import pandas as pd
start = time.clock()

fname = sys.argv[2]

layersets,flags = pm.laminar_components(fname)

inhFiles = glob.glob('./data/'+fname+'*1.0*5000*.gdf')
excFiles = glob.glob('./data/'+fname+'*-18*5000*.gdf')
extFiles = glob.glob('./data/'+fname+'*1.0*-18.*.gdf')
inhFiles.sort(),excFiles.sort(),extFiles.sort()

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
    print q[7:]
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
        # ffnames[i],ff[i]

df = pd.DataFrame(ff,index=ffnames,columns=['Fano Factor'])
df.to_excel("./fano/fano"+fname+sys.argv[1]+".xlsx")

#print "done!"
end = time.clock()
#print "Runtime: %.5s seconds" %(end - start)