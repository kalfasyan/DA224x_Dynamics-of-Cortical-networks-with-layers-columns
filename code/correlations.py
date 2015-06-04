# run correlations.py 100
import glob
import numpy as np
import parameters_v1 as pm
from scipy.stats.stats import pearsonr
import sys
import pylab as py

fname = sys.argv[1] #for example: '101'
sd_filename = glob.glob('./data/'+fname+'*1.0_-17*.gdf')[0]


""" RETRIEVING LAMINAR COMPONENTS, NAMES """
layersets,flags = pm.laminar_components(fname)

""" loading file, collecting neuron ids and creating range for histogram """
xx = np.loadtxt(sd_filename)
n_id = xx[:,0]
ed = np.arange(100,2e+3,5.)
pop_act,act_names = [],[]


""" CREATING POPULATION ACTIVITIES """
for i in range(len(layersets)):
    cor1 = np.zeros((len(xx[:,:]),2))
    for j in range(len(xx[:,1])):
        if n_id[j] in layersets[i]:
            cor1[j] = xx[:,:][j]
    pop_act.append(np.histogram(cor1[:,1],ed)[0])
    act_names.append(flags[i])


""" CREATING THE CORRELATION MATRIX """
pearson_mat = np.zeros((len(pop_act),len(pop_act)))
txt = ""
for i in range(len(pop_act)):
    #txt = act_names[i] + " "
    for j in range(len(pop_act)):
        pearson_mat[j][i] = pearsonr(pop_act[i],pop_act[j])[0]
        #print txt + act_names[j]


""" PLOTTING """
fig,ax = py.subplots()

# turn off the frame
ax.set_frame_on(False)
# Format
fig = py.gcf()
fig.set_size_inches(2*len(pop_act), len(pop_act))
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(pearson_mat.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(pearson_mat.shape[1]) + 0.5, minor=False)
# setting the labels
ax.set_xticklabels(act_names, minor=False)
ax.set_yticklabels(act_names,minor=False)

im = ax.pcolor(pearson_mat,cmap=py.cm.Blues)
for y in range(pearson_mat.shape[0]):
    for x in range(pearson_mat.shape[1]):
        py.text(x + 0.5, y + 0.5, '%.3f' % pearson_mat[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
#fig.colorbar(im)
ax.grid(False)
ax = py.gca()
#py.show()
py.savefig('./figures/correlation'+sd_filename[7:]+".", bbox_inches='tight')