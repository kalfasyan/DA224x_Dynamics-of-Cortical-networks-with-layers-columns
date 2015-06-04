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

fname = '100'#sys.argv[2]

""" RETRIEVING LAMINAR COMPONENTS """
layersets,flags = pm.laminar_components(fname)

""" collecting spike_detector files for the exc/inh/ext tests """
inhFiles = glob.glob('./data/'+fname+'*1.0*5000*.gdf')
excFiles = glob.glob('./data/'+fname+'*-18*5000*.gdf')
extFiles = glob.glob('./data/'+fname+'*1.0*-18.*.gdf')
inhFiles.sort(),excFiles.sort(),extFiles.sort()

user_input = {"inhFiles": inhFiles,
              "excFiles": excFiles,
              "extFiles": extFiles}
filenames = user_input['inhFiles'] # User gives e.g inhFiles for inhibitory test files

bins = 5.
ed = np.arange(100,2e+3,bins)
ff,ffnames = [],[]
# This creates ff[i] = fano factors OR/AND ffnames[i] for the labels
for q in filenames:
    sd_filename = q
    print q[7:]
    xx = np.loadtxt(sd_filename)
    n_id = xx[:,0]
    t_id = xx[:,1] # times of spike events
    for i in range(len(layersets)):
        cor1 = np.zeros((len(xx[:,:]),2))
        for j in range(len(xx[:,1])):
            if n_id[j] in layersets[i] and t_id[j] > 300.:
                cor1[j] = xx[:,:][j]
        pop_x = np.histogram(cor1[:,1],ed)
        pop_hh = pop_x[0]/(bins*1e-3)/pm.nrns
        ff.append( np.var(pop_hh)/np.mean(pop_hh))
        ffnames.append(flags[i])
        # ffnames[i],ff[i]

df = pd.DataFrame(ff,index=ffnames,columns=[filenames])
df.to_excel("./fano/fano"+fname+sys.argv[1]+".xlsx")

#print "done!"
end = time.clock()
#print "Runtime: %.5s seconds" %(end - start)